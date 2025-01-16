import torch
import pandas as pd
import os, time
import numpy as np
from PIL import Image
import open3d as o3d
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import resnet18, ResNet18_Weights

# Make sure you have the correct import for PointPillars
from model.pointpillars import PointPillars

start_time = time.time()

class ImageFeature:
    def __init__(self, in_csv_path):
        self.in_csv_path = in_csv_path
        self.model = self.resnet18()
        self.temp_csv = self.image_inference(self.model, self.in_csv_path)

    def resnet18(self):
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()  # Remove the classification layer
        model.eval()
        return model

    def image_inference(self, model, in_csv_path):
        in_df = pd.read_csv(in_csv_path)
        column_name = "IMG filename"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Determine the output CSV path
        parent_folder = os.path.dirname(in_csv_path)
        output_csv_path = os.path.join(parent_folder, "temp_csv.csv")

        # Create the folder for NPY files
        output_subfolder = os.path.join(parent_folder, "images_npy")
        os.makedirs(output_subfolder, exist_ok=True)

        # Iterate through the rows in the DataFrame
        for index, row in in_df.iterrows():
            image_path = row[column_name]

            # Get the base name of the file (without extension)
            base_name = os.path.splitext(os.path.basename(image_path))[0]

            input_tensor = self.image_transform(image_path)

            # Perform inference
            with torch.no_grad():
                features = model(input_tensor)
                # features is shape [1, 512] for resnet18’s penultimate layer
                features = features[0]
                output = features.cpu().numpy()

            # Save the output to an NPY file
            npy_file_name = f"{base_name}.npy"  # Same name as input, but with .npy extension
            npy_file_path = os.path.join(output_subfolder, npy_file_name)
            np.save(npy_file_path, output)

            # Update the DataFrame with the new path
            in_df.at[index, column_name] = npy_file_path

        # Save the updated DataFrame to the output CSV
        in_df.to_csv(output_csv_path, index=False)

        print(f"Updated CSV saved at: {output_csv_path}")
        return output_csv_path

    def image_transform(self, image_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.ToTensor(),          # PIL Image to Tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]    # ImageNet std
            )
        ])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
        img = transform(img)
        img = img.unsqueeze(0).to(device)
        return img


class PillarFeatureExtractor(nn.Module):
    """
    Wraps the submodules of PointPillars model so that forward() returns a feature vector (e.g. 512D).
    """
    def __init__(self, original_model: PointPillars, fc_dim=512):
        super().__init__()
        self.pillar_layer = original_model.pillar_layer
        self.pillar_encoder = original_model.pillar_encoder
        self.backbone = original_model.backbone
        self.neck = original_model.neck

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # The '384' in the Linear below corresponds to the channel dimension
        self.fc = nn.Linear(384, fc_dim)

    def forward(self, batched_pts):
        """
        :param batched_pts: list of Tensors (length = batch_size)
        """
        # Pillar features
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)
        pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)

        # Backbone + Neck
        xs = self.backbone(pillar_features)
        feats = self.neck(xs)

        # Pool down to a single vector per sample
        pooled = self.pool(feats)  # shape [B, C, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)  # shape [B, C]

        out_512 = self.fc(pooled)  # shape [B, fc_dim]
        return out_512


class PCDFeature:
    def __init__(self, temp_csv_path):
        self.temp_csv_path = temp_csv_path

        # Load model (PointPillars)
        self.model = self.load_pointpillars_model()
        self.model.eval()
        self.feature_csv = self.pcd_inference(self.model, self.temp_csv_path)

    def load_pointpillars_model(self):
        """
        Instantiates and loads the PointPillars model.
        """
        pointpillars = PointPillars(
            nclasses=3,
            voxel_size=[0.16, 0.16, 4],
            point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
            max_num_points=32,
            max_voxels=(16000, 40000)
        )
        checkpoint_path = "/home/ws2/Documents/student_project_WS_24_25/Student_Project_WS24/npy data/pre_trained/epoch_160.pth"
        pointpillars.load_state_dict(torch.load(checkpoint_path, weights_only=True), strict=False)
        return pointpillars

    def load_xyz_intensity_from_pcd(self, pcd_path):
        """
        Reads .pcd with x,y,z + intensity embedded in colors.
        Returns a NumPy array of shape [N, 4].
        """
        pcd = o3d.io.read_point_cloud(pcd_path)
        xyz = np.asarray(pcd.points)            # (N, 3)
        colors = np.asarray(pcd.colors)         # (N, 3) but (R=G=B => intensity)

        # Recover intensity
        intensity = colors[:, 0]               # or an average if needed
        intensity *= 100.0
        points_np = np.hstack([xyz, intensity.reshape(-1, 1)])
        return points_np

    def pcd_inference(self, model, temp_csv_path):
        # Load the input CSV
        in_df = pd.read_csv(temp_csv_path)
        column_name = "PCD filename"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Create the feature-extraction wrapper from loaded model
        feature_extractor = PillarFeatureExtractor(model).to(device)

        # Determine the output CSV path
        parent_folder, _ = os.path.split(temp_csv_path)
        parent_folder_strip = parent_folder.rstrip('/')
        rosbag_name = os.path.basename(parent_folder_strip)
        filename = f"{rosbag_name}_feat_data.csv"
        output_csv_path = os.path.join(parent_folder, filename)

        # Create the folder for NPY files
        output_subfolder = os.path.join(parent_folder, "pcd_npy")
        os.makedirs(output_subfolder, exist_ok=True)

        # Iterate through the rows in the DataFrame
        for index, row in in_df.iterrows():
            pcd_path = row[column_name]

            # Get the base name of the file (without extension)
            base_name = os.path.splitext(os.path.basename(pcd_path))[0]

            points = self.load_xyz_intensity_from_pcd(pcd_path)

            # Convert to a list of 1 Tensor for the batch
            input_tensor = [torch.from_numpy(points).float().to(device)]

            with torch.no_grad():
                # Pass it to the feature extractor
                features = feature_extractor(input_tensor)  # shape [1, fc_dim]
                features = features[0]                     # shape [fc_dim]

                output = features.cpu().numpy()

            # Save the output to an NPY file
            npy_file_name = f"{base_name}.npy"
            npy_file_path = os.path.join(output_subfolder, npy_file_name)
            np.save(npy_file_path, output)

            # Update the DataFrame with the new path
            in_df.at[index, column_name] = npy_file_path

        # Save the updated DataFrame to the output CSV
        in_df.to_csv(output_csv_path, index=False)
        print(f"\nUpdated CSV saved at: {output_csv_path}\n")

        return output_csv_path


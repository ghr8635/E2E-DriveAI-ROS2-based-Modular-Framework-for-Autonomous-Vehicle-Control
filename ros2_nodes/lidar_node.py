import rclpy
from rclpy.node import Node
from obdreader.msg import SynchronizedRawData  # Replace with your synchronized message type
from sensor_msgs.msg import PointCloud2
from obdreader.msg import Feature
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from pathlib import Path
from scripts import pointpillars
from scripts import voxel_module  # Import your separate voxelization module


class PillarFeatureExtractor(nn.Module):
    """
    Wraps the submodules of PointPillars model so that forward() returns a feature vector (e.g. 512D).
    """
    def __init__(self, original_model: pointpillars.PointPillars, fc_dim=512):
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


class MLLidarPointPillarsNode(Node):
    def __init__(self):
        super().__init__('ml_lidar_pointpillars_node')
        
        # Subscribe to the synchronized data topic
        self.subscription = self.create_subscription(
            SynchronizedRawData,
            '/synchronized_raw_data_with_features',
            self.point_cloud_callback,
            10)

        self.publisher = self.create_publisher(Feature, 'combined_features', 10)

        # Load the PointPillars model
        model_path = Path("pointpillar/pre_trained/epoch_160.pth")
        self.model = self.load_pointpillars_model(model_path)
        self.model.eval()  # Set model to evaluation mode

        # Initialize voxelization parameters using the separate module
        self.voxelizer = voxel_module.HardVoxelization(
            voxel_size=[0.16, 0.16, 4],
            point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
            max_num_points=32,
            max_voxels=(16000, 40000)
        )

        self.get_logger().info("ML LiDAR Node with PointPillars model initialized")

    def load_pointpillars_model(self, model_path):
        # Initialize the PointPillars model
        model = pointpillars.PointPillars(
            nclasses=3,
            voxel_size=[0.16, 0.16, 4],
            point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
            max_num_points=32,
            max_voxels=(16000, 40000)
        )

        model_state_dict = torch.load(model_path, map_location=torch.device('cuda'), weights_only=True)

        # Load the filtered state dict into the model (without the head weights)
        model.load_state_dict(model_state_dict, strict=False)

        return model
    
    def point_cloud_callback(self, msg):
        # Extract the PointCloud2 message from the synchronized message
        point_cloud_msg = msg.pointcloud

        # Convert PointCloud2 message to numpy array
        pcd_points = self.convert_pointcloud2_to_array(point_cloud_msg)

        # Preprocess point cloud to fit PointPillars input format
        input_tensor = self.load_xyz_intensity_from_pcd(pcd_points)
        self.get_logger().info(f"Input tensor shape: {input_tensor.shape}")

        # Run inference with the PointPillars model
        features_pcd = self.run_pointpillars_inference(input_tensor)
        self.get_logger().info(f"PointPillars features shape: {features_pcd.shape}")

        # Extract the image features from the synchronized message
        features_img = msg.features  # Assuming features from image are already in the message
        features_img = torch.tensor(features_img, dtype=torch.float32).to('cuda')

        # Combine features from image and point cloud into a single tensor
        combined_features = torch.cat((features_img, features_pcd), dim=0)
        self.get_logger().info(f"Combined features shape: {combined_features.shape}")

        # Handle the result (e.g., publish detections, log output)
        self.handle_result(combined_features)

    def convert_pointcloud2_to_array(self, cloud_msg):
        # Convert ROS PointCloud2 message to an array (XYZ + intensity)
        point_cloud = np.frombuffer(cloud_msg.data, dtype=np.float32).reshape(-1, 4)
        return point_cloud

    def load_xyz_intensity_from_pcd(self, point_cloud):
        """
        Reads .pcd with x,y,z + intensity embedded in colors.
        Returns a PyTorch tensor of shape [1, N, 4], where N is the number of points.
        """
        
        # Directly create the tensor for xyz and intensity
        xyz_tensor = torch.tensor(point_cloud[:, :3], dtype=torch.float32).to('cuda')  # Shape: (N, 3)
        intensity_tensor = torch.tensor(point_cloud[:, 3], dtype=torch.float32).reshape(-1, 1).to('cuda')  # Shape: (N, 1)

        # Concatenate xyz and intensity tensors directly
        points_tensor = torch.cat([xyz_tensor, intensity_tensor], dim=1)  # Shape: (N, 4)

        # Add batch dimension to make it [1, N, 4]
        points_tensor = points_tensor.unsqueeze(0)  # Shape: (1, N, 4)

        return points_tensor
    def run_pointpillars_inference(self, input_tensor):
        # Run the model inference
        feature_extractor = PillarFeatureExtractor(self.model).to("cuda")
        with torch.no_grad():
            features = feature_extractor(input_tensor)
        output = features[0]  # Shape will be [512]

        return output

    def handle_result(self, result):
        # Flatten the tensor and convert it to a list
        features_list = result.reshape(-1).tolist()

        # Create a Feature message
        feature_msg = Feature()
        feature_msg.features = features_list

        # Publish the feature message
        self.publisher.publish(feature_msg)
        self.get_logger().info(f"Published feature message with {len(features_list)} features")

def main(args=None):
    rclpy.init(args=args)
    node = MLLidarPointPillarsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from obdreader.msg import Feature
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from pathlib import Path
from scripts import pointpillars
from scripts import voxel_module  # Import your separate voxelization module


class MLLidarPointPillarsNode(Node):
    def __init__(self):
        super().__init__('ml_lidar_pointpillars_node')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/velodyne_points',
            self.point_cloud_callback,
            10)

        self.publisher = self.create_publisher(Feature, '/pp_features', 10)

        # Load the PointPillars model
        model_path = Path("/home/linux1/ros2_new_ws/src/ams_motor_drive/scripts/epoch_160.pth")
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

        model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        # Load the filtered state dict into the model (without the head weights)
        model.load_state_dict(model_state_dict, strict=False)

        # Replace the head layer with Identity
        model.neck = nn.Identity()
        model.head = nn.Identity()

        return model
    
    def point_cloud_callback(self, msg):
        # Convert PointCloud2 message to numpy array
        pcd_points = self.convert_pointcloud2_to_array(msg)

        # Preprocess point cloud to fit PointPillars input format
        input_tensor = self.preprocess_point_cloud(pcd_points)
        print(input_tensor.shape)

        # Run inference with the PointPillars model
        features = self.run_pointpillars_inference(input_tensor)
        B, C, H, W = features.shape  # B=1, C=256, H=62, W=54

        # Reshape PointPillars features to [Batch, Seq_Len, Channels]
        pointpillars_features_flat = features.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [1, 3348, 256]
        print(pointpillars_features_flat.shape)

        # Handle the result (e.g., publish detections, log output)
        self.handle_result(pointpillars_features_flat)

    def convert_pointcloud2_to_array(self, cloud_msg):
    # Convert ROS PointCloud2 message to an array (XYZ + intensity)
        point_cloud = np.frombuffer(cloud_msg.data, dtype=np.float32).reshape(-1, 4)
        return point_cloud

    def clean_and_inspect_data(self, points):

        # Remove NaN/Inf
        points = points[np.isfinite(points).all(axis=1)]

        # Remove extremely large/small values
        points[np.abs(points) > 1e6] = np.nan
        points = points[np.isfinite(points).all(axis=1)]

        # Final inspection
        print(f"Final Min value: {np.min(points)}, Max value: {np.max(points)}")
        if points.shape[0] == 0:
            raise ValueError("No valid points remaining after filtering!")

        return points

    
    def preprocess_point_cloud(self, pcd_points):
        pcd_points = self.clean_and_inspect_data(pcd_points)

        pcd_points = pcd_points[:, :3]  # Shape: (N, 3)
        print(f"Filtered points shape: {pcd_points.shape}")

        # For example, voxel size (you may need to tune this)
        voxel_size = 0.2
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_points)
        downsampled_pcd = pcd.voxel_down_sample(voxel_size)

        # Convert the downsampled points to numpy array
        downsampled_points = np.asarray(downsampled_pcd.points)  # Shape: (N', 3)

        # Convert to torch tensor and add batch dimension
        input_tensor = torch.tensor(downsampled_points, dtype=torch.float32).unsqueeze(0)

        return input_tensor


    def run_pointpillars_inference(self, input_tensor):
        # Prepare the input for the model
        with torch.no_grad():
            output = self.model(input_tensor)
            output = output[-1]

        # Process the model output as required
        return output

    def handle_result(self, result):
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

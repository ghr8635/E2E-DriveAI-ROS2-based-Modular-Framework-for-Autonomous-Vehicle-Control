#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from obdreader.msg import SynchronizedRawData  # Define this custom message type
from collections import deque
from cv_bridge import CvBridge
import cv2
import torch
import torchvision.transforms as transforms
from pathlib import Path
from torchvision import models

class RawDataSynchronizer(Node):
    def __init__(self):
        super().__init__('raw_data_synchronizer')

        # Buffer for image messages (fixed size)
        self.image_buffer = deque(maxlen=10)  # Adjust size as needed

        # Subscribers for raw image and PCD data
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.create_subscription(PointCloud2, '/velodyne_points', self.pcd_callback, 10)

        # Publisher for combined synchronized data
        self.pub = self.create_publisher(SynchronizedRawData, '/synchronized_raw_data_with_features', 10)

        self.bridge = CvBridge()

        # Load your custom model
        self.model = self.load_trained_model()
        self.model.eval()  # Set model to evaluation mode

        # Define the image transformation pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.ToTensor(),          # PIL Image to Tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]    # ImageNet std
            )
        ])

        # Log initialization
        self.get_logger().info("Raw data synchronizer with feature extraction initialized.")

    def load_trained_model(self):
        # Initialize ResNet-18 architecture
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        model.fc = torch.nn.Identity()  # Remove the classification layer
        # Move the model to CUDA
        model.to(device="cuda")
        return model

    def image_callback(self, msg):
        """Callback for Image messages."""
        # Update timestamp to current time and add to buffer
        current_time = self.get_clock().now().to_msg()
        msg.header.stamp = current_time
        self.image_buffer.append(msg)

    def pcd_callback(self, msg):
        """Callback for PCD messages."""
        if not self.image_buffer:
            return  # No images to synchronize with

        # Update timestamp to current time
        current_time = self.get_clock().now().to_msg()
        msg.header.stamp = current_time

        # Synchronize with the most recent image in the buffer
        latest_image = self.image_buffer[-1]  # Get the most recent image

        # Extract features from the synchronized image
        features = self.extract_features(latest_image)

        # Create and publish synchronized data with features
        synced_data = SynchronizedRawData()
        synced_data.header.stamp = current_time  # Current time for the synchronized message
        synced_data.header.frame_id = "synchronized_raw_data_with_image_features"
        #synced_data.image = latest_image
        synced_data.pointcloud = msg
        synced_data.features = features  # Add extracted features to the message

        self.pub.publish(synced_data)

        # Clear the image buffer after synchronization
        self.image_buffer.clear()

        # Log the synchronized message
        self.get_logger().info(f"Published synchronized message with features at time: {current_time.sec}.{current_time.nanosec}")

    def extract_features(self, image_msg):
        """Extract features from an image message."""
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        # Preprocess the image for the custom model
        processed_image = self.preprocess_image(cv_image)
        # Move the processed image to CUDA
        processed_image = processed_image.to(device="cuda")

        # Perform inference
        with torch.no_grad():
            output = self.model(processed_image)
        
        # Flatten the tensor and convert it to a list
        features_list = output[0].view(-1).tolist()
        print(f"Type of features_list: {type(features_list)}")
        return features_list

    def preprocess_image(self, image):
        """Preprocess image for the model."""
        # Convert the OpenCV image (BGR) to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply the transformations
        transformed_image = self.transform(image_rgb)
        transformed_image = transformed_image.unsqueeze(0)  # Add batch dimension
        return transformed_image

def main(args=None):
    rclpy.init(args=args)

    raw_data_synchronizer = RawDataSynchronizer()

    try:
        raw_data_synchronizer.get_logger().info("Starting Raw Data Synchronizer Node...")
        rclpy.spin(raw_data_synchronizer)
    except KeyboardInterrupt:
        raw_data_synchronizer.get_logger().info("Shutting down Raw Data Synchronizer Node.")
    except Exception as e:
        raw_data_synchronizer.get_logger().error(f"Unexpected error: {e}")
    finally:
        raw_data_synchronizer.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from obdreader.msg import SynchronizedRawData  # Define this custom message type
from obdreader.msg import Feature
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

        # Publisher for synchronized raw data
        self.pub = self.create_publisher(SynchronizedRawData, '/synchronized_raw_data', 10)

        # Log initialization
        self.get_logger().info("Raw data synchronizer node initialized.")

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

        # Create and publish synchronized data
        synced_data = SynchronizedRawData()
        synced_data.header.stamp = current_time  # Current time for the synchronized message
        synced_data.header.frame_id = "synchronized_raw_data"
        synced_data.image = latest_image
        synced_data.pointcloud = msg

        self.pub.publish(synced_data)

        # Clear the image buffer after synchronization
        self.image_buffer.clear()

        # Log the synchronized message
        self.get_logger().info(f"Published synchronized message at time: {current_time.sec}.{current_time.nanosec}")

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

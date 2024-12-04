#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image, PointCloud2
from obdreader.msg import SynchronizedRawData  # Define this custom message type

class RawDataSynchronizer(Node):
    def __init__(self):
        super().__init__('raw_data_synchronizer')

        # Subscribers for raw image and PCD data
        self.image_sub = Subscriber(self, Image, '/camera/image_raw')
        self.pcd_sub = Subscriber(self, PointCloud2, '/velodyne_points')

        # Synchronizer for approximate time synchronization
        self.sync = ApproximateTimeSynchronizer([self.image_sub, self.pcd_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.callback)

        # Publisher for synchronized raw data
        self.pub = self.create_publisher(SynchronizedRawData, '/synchronized_raw_data', 10)

        self.get_logger().info("Raw data synchronizer node initialized.")

    def callback(self, image, pcd):
        """
        Callback function for synchronized raw image and PCD data.
        Combines them and publishes as a single message.
        """
        self.get_logger().info("Synchronized raw data received.")

        # Create a message for synchronized raw data
        synced_data = SynchronizedRawData()
        synced_data.header.stamp = self.get_clock().now().to_msg()  # Set the current timestamp
        synced_data.header.frame_id = "synchronized_raw_data"

        # Store synchronized image and PCD data
        synced_data.image = image
        synced_data.pointcloud = pcd

        # Publish the synchronized data
        self.pub.publish(synced_data)

def main(args=None):
    rclpy.init(args=args)

    raw_data_synchronizer = RawDataSynchronizer()

    try:
        rclpy.spin(raw_data_synchronizer)
    except KeyboardInterrupt:
        raw_data_synchronizer.get_logger().info("Shutting down Raw Data Synchronizer Node.")
    finally:
        raw_data_synchronizer.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

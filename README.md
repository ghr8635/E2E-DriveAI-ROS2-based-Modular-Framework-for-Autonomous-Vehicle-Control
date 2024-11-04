**Project Overview**
This project, E2E DriveAI, is a ROS2-based modular framework designed to enable end-to-end autonomous vehicle control using raw sensor data. It integrates exteroceptive sensors, including cameras, LiDAR, and on-board diagnostic (OBD) data, to develop a comprehensive autonomous driving solution.

The framework includes the following components:

**Sensor Data Acquisition and Visualization:** Extract and visualize camera, LiDAR, and OBD data in ROS2.

**Deep Learning for Control Prediction:** Develop and train a deep learning model for predicting control inputs based on camera and LiDAR data. The architecture combines a ResNet-based feature extraction model with a Transformer network to handle temporal dependencies and refine control predictions.

**Vehicle Actuation and Control:** Implement a low-level controller to actuate brake, acceleration, and steering based on predicted inputs.

**Evaluation Scenarios:** Single-lane and multi-lane scenarios with predefined behaviors to evaluate collision avoidance, braking, and lane-changing maneuvers.

**Goals**
1. Develop and integrate data loaders and a training pipeline.
2. Build and train a deep learning architecture combining ResNet for feature detection and Transformers for sequence prediction to generate accurate control inputs.
3. Implement and evaluate the framework in both single-lane and two-lane autonomous driving scenarios.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from obdreader.msg import Feature
from cv_bridge import CvBridge
import cv2
import torch
import torchvision.transforms as transforms
from pathlib import Path
from torchvision import models

class MLImageNode(Node):
    def __init__(self):
        super().__init__('ml_image_node')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)

        self.publisher = self.create_publisher(Feature, '/image_features', 10)
        self.bridge = CvBridge()

        # Define the path to your custom model
        model_path = Path("/home/linux1/ros2_new_ws/src/ams_motor_drive/scripts/feature_extractor_resnet18.pth")

        # Load your custom model (adjust the model structure if necessary)
        self.model = self.load_trained_model(model_path)
        self.model.eval()  # Set model to evaluation mode
        
        # Define the image transformation pipeline (resize, normalization, etc.)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # Adjust to your model's input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.get_logger().info("ML Image Node with Custom Model initialized")
    
    def load_trained_model(self, model_path):
        # Initialize ResNet-18 architecture
        model = models.resnet18()
        model.fc = torch.nn.Linear(model.fc.in_features, 10)

        state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        model.load_state_dict(state_dict)

        model.fc = torch.nn.Identity()
        return model


    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Preprocess the image for the custom model
        processed_image = self.preprocess_image(cv_image)

        # Perform inference
        features = self.run_inference(processed_image)
        H = 62          #these values are obtaiend from pointpillar inference, in order to match 
        W = 54
        result = features.unsqueeze(1).repeat(1, H * W, 1)  # [1, 3348, 512]

        print(result.shape)

        # Handle the result (e.g., log or publish it)
        self.handle_result(result)

    def preprocess_image(self, image):
        # Convert the OpenCV image (BGR) to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply the transformations
        transformed_image = self.transform(image_rgb)
        transformed_image = transformed_image.unsqueeze(0)  # Add batch dimension
        return transformed_image

    def run_inference(self, image):
        # Run the custom model on the input image
        with torch.no_grad():
            output = self.model(image)
        # Convert the output to a usable format (e.g., feature vector or classification result)
        result = output  # Process according to your model's output format
        return result

    def handle_result(self, result):
        # Flatten the tensor and convert it to a list
        features_list = result.view(-1).tolist()

        # Create a Feature message
        feature_msg = Feature()
        feature_msg.features = features_list

        # Publish the feature message
        self.publisher.publish(feature_msg)
        self.get_logger().info(f"Published feature message with {len(features_list)} features")


def main(args=None):
    rclpy.init(args=args)
    node = MLImageNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

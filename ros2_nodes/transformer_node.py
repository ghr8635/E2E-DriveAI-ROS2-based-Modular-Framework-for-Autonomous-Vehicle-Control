import rclpy
from rclpy.node import Node
from obdreader.msg import Feature  # Replace with your actual message type for combined features
import torch
from pathlib import Path
from collections import deque
from scripts.transformer_only_enc import Transformer

class TransformerNode(Node):
    def __init__(self):
        super().__init__('transformer_node')

        # Set device to CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Subscriber to the topic where combined features are published
        self.subscription = self.create_subscription(
            Feature,
            '/combined_features',  # The topic where previous node publishes combined features
            self.features_callback,
            10
        )

        # Buffer to store the last 25 messages
        self.buffer = deque(maxlen=25)  # Deque will automatically discard the oldest element when the buffer is full

        # Load the Transformer model
        model_path = Path("transformer/pre_trained/driving_transformer.pt")
        self.model = self.load_transformer_model(model_path)
        self.model.eval().to(self.device)  # Set model to evaluation mode and move to CUDA

        self.get_logger().info("Transformer Node initialized")

    def load_transformer_model(self, model_path):
        # Load the saved Transformer model
        model = Transformer(
            input_dim=1024,
            d_model=512,
            nheads=8,
            num_encoder_layers=8,
            dim_feedforward=1024,
            dropout=0.1,
            future_len=15,   # predict 15 timesteps
            output_dim=2     # speed + steer
        )
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model

    def features_callback(self, msg):
        # Convert received features to tensor and move to device
        new_features = torch.tensor(msg.features, dtype=torch.float32).to(self.device)
        print(new_features.shape)

        # Add the new features to the buffer
        self.buffer.append(new_features)

        # Once we have 25 messages in the buffer, run inference
        if len(self.buffer) == 25:
            self.run_inference()

    def run_inference(self):
        # Stack the 25 messages into a single sample of shape [25, 1024] and move to device
        stacked_features = torch.stack(list(self.buffer), dim=0).unsqueeze(0).to(self.device)  # Shape: [1, 25, 1024]


        # Perform inference on the stacked features
        with torch.no_grad():
            output = self.model(stacked_features)  # Run inference with the stacked sample
            print(output.shape)

        # Publish the output from the transformer
        # self.handle_result(output)

    # def handle_result(self, result):
        # Convert the result to a list and prepare the message
        # result_list = result.view(-1).tolist()
        # output_msg = TransformerOutput()
        # output_msg.output = result_list

        # # Publish the output message
        # self.publisher.publish(output_msg)
        # self.get_logger().info(f"Published Transformer output with {len(result_list)} elements")


def main(args=None):
    rclpy.init(args=args)
    node = TransformerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

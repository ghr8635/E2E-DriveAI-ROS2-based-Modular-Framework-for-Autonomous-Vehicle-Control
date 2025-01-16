import torch, os, json, cv2
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
from scipy.stats import mode
import open3d as o3d
from FeatureExtract import ImageFeature, PCDFeature


class CustomDataset(Dataset):
    def __init__(self, masterinfo_path, config_path):
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)

        self.past = []
        self.future = []

        self.req_in_freq = self.config["req_in_freq"]
        self.req_op_freq = self.config["req_op_freq"]
        self.observation_window = self.config["observation_window"]
        self.control_window = self.config["control_window"]
        
        self.speed_min = 0.0
        self.speed_max = 30.0
        self.steer_min = -100.0
        self.steer_max = 100.0

        self.past, self.future = self.sample_data(
            masterinfo_path,
            self.req_in_freq,
            self.req_op_freq,
            self.observation_window,
            self.control_window
        )

    def __len__(self):
        return len(self.past)

    def __getitem__(self, idx):
        """
        Retrieve an item by index. Also extract PCD and IMG data from the path in variable: Past
        """

        num_entries_per_sample = self.observation_window * self.req_in_freq

        # Initialize img and pcd as 2D NumPy arrays for efficient storage
        img = np.empty((num_entries_per_sample, 512), dtype=np.float32)
        pcd = np.empty((num_entries_per_sample, 512), dtype=np.float32)

        img_feature_paths = [str(self.past[idx][j][1]) for j in range(num_entries_per_sample)]
        pcd_feature_paths = [str(self.past[idx][j][0]) for j in range(num_entries_per_sample)]
        
        # Extract features
        img = self.getfeature(img_feature_paths)  # Assign directly to pre-allocated array
        pcd = self.getfeature(pcd_feature_paths)  # Assign to pre-allocated list

        # Convert data to appropriate types
        img = torch.tensor(img, dtype=torch.float32)  # Convert images to a Torch tensor
        pcd = torch.tensor(pcd, dtype=torch.float32)  # Ensure proper indentation with spaces

        # Extract the train/test label before using it
        train_test_label = int(float(self.past[idx][0][3]))

        # Prepare x dictionary
        x = {
            "Image": img,
            "PCD": pcd,
            "img_paths": img_feature_paths,
            "pcd_paths": pcd_feature_paths
        }

        # Prepare y dictionary (speed and steer data are stored in self.future[idx])
        # Speed in [0..30] -> [0..1]
        # Steer in [-100..100] -> [-1..1]
        
        speeds = torch.tensor(self.future[idx][:, 0], dtype=torch.float32)
        steers = torch.tensor(self.future[idx][:, 1], dtype=torch.float32)
                
        speeds_norm = (speeds - self.speed_min) / (self.speed_max - self.speed_min)  # -> [0..1]
        steers_norm = 2.0 * (steers - self.steer_min) / (self.steer_max - self.steer_min) - 1.0  
        # => [-1..1] (shifting after scaling to 0..1) 
        # -------------------------------------------------------------

        y = {
            "Speed": speeds_norm,
            "Steer": steers_norm
        }
        
        return x, y, train_test_label

    def estimate_frequency(self, df, column_name):
        timestamps = df[column_name].dropna().values        
        time_deltas = np.diff(timestamps)             # Compute differences between consecutive timestamps

        # Filter out anomalies using median absolute deviation (MAD)
        median_delta = np.median(time_deltas)
        mad = np.median(np.abs(time_deltas - median_delta))
        threshold = 3 * mad  # Adjust as needed for sensitivity
        filtered_deltas = time_deltas[np.abs(time_deltas - median_delta) <= threshold]
        frequency = 1 / np.median(filtered_deltas)  # Use median of filtered deltas for robust estimation
        return frequency

    def sample_data(self, masterinfo_path, req_in_freq, req_op_freq, observation_window, control_window):

        masterinfo = pd.read_csv(masterinfo_path)

        # Iterate through each row in the masterinfo file
        print('Iterating through rows of Master csv file')

        for index, row in masterinfo.iterrows():

            rosbag_prefix = row['ROS bags']  # To save sample names with indices
            if isinstance(rosbag_prefix, float) and np.isnan(rosbag_prefix):
                break
            else:
                train_test = row['TrainTest']  # To know if this scenario is for training or test
                in_csv_path = row['IN']        # Path to IN CSV
                op_csv_path = row['OP']        # Path to OP CSV
                in_sample_path = f"{os.path.dirname(in_csv_path)}/IN samples"
                op_sample_path = f"{os.path.dirname(op_csv_path)}/OP samples"
                os.makedirs(in_sample_path, exist_ok=True)
                os.makedirs(op_sample_path, exist_ok=True)

                in_df = pd.read_csv(in_csv_path)
                op_df = pd.read_csv(op_csv_path)

                est_in_freq = self.estimate_frequency(in_df, 'Timestamp')
                est_op_freq = self.estimate_frequency(op_df, 'Timestamp')

                if req_in_freq > round(est_in_freq):
                    print(
                        f"Expected frequency of past data: {req_in_freq} is more than actual frequency of data: {est_in_freq}")
                    break

                if req_op_freq > round(est_op_freq):
                    print(
                        f"Expected frequency of future data: {req_op_freq} is more than actual frequency of data: {est_op_freq}")
                    break

                # Calculate step sizes for sliding windows
                in_step = int(round(est_in_freq / req_in_freq)) if est_in_freq and req_in_freq else 1
                op_step = int(round(est_op_freq / req_op_freq)) if est_op_freq and req_op_freq else 1

                in_required_rows = int(observation_window * req_in_freq)
                op_required_rows = int(control_window * req_op_freq)

                sample_index = 1
                start_row = 0

                while start_row + in_required_rows < len(in_df):

                    in_sample_indices = range(start_row, start_row + in_required_rows * in_step, in_step)

                    if max(in_sample_indices, default=0) >= len(in_df) or len(
                            in_df.iloc[in_sample_indices]) < in_required_rows:
                        print(f"Done sampling IN after {sample_index - 1} samples. Stopping...")
                        break

                    in_sample = in_df.iloc[in_sample_indices] if max(in_sample_indices, default=0) < len(
                        in_df) else in_df.iloc[start_row:]

                    # Find the closest timestamp in OP CSV for the last timestamp in IN sample
                    last_timestamp = in_sample['Timestamp'].iloc[-1]
                    op_closest_idx = op_df[op_df['Timestamp'] > last_timestamp]['Timestamp'].idxmin()

                    # Stops the loop if it cannot find the closest op timestamp value beyond the given timestamp
                    if pd.isna(op_closest_idx):
                        print(f"Done sampling OP after {sample_index - 1} samples. No valid OP timestamp found.")
                        break

                    # Get the next required number of rows from OP with step
                    op_sample_indices = range(op_closest_idx, op_closest_idx + op_required_rows * op_step, op_step)

                    if max(op_sample_indices, default=0) >= len(op_df) or len(
                            op_df.iloc[op_sample_indices]) < op_required_rows:
                        # print(f"Done sampling OP after {sample_index - 1} samples. Stopping...")
                        break

                    op_sample = op_df.iloc[op_sample_indices] if max(op_sample_indices, default=0) < len(
                        op_df) else op_df.iloc[op_closest_idx:]

                    in_sample.insert(3, 'ROS_bag', rosbag_prefix)            # Add 'ROS_bag' as the 3rd column
                    in_sample.insert(4, 'TrainTest', str(train_test))       # Add 'TrainTest' as the 4th column

                    # Print or save the sampled data
                    # (commented out for brevity)
                    # in_sample.to_csv(f"{in_sample_path}/{rosbag_prefix}_IN_sample_{sample_index}.csv", index=False)
                    # op_sample.to_csv(f"{op_sample_path}/{rosbag_prefix}_OP_sample_{sample_index}.csv", index=False)

                    self.past.append(in_sample.drop(columns=['Timestamp']).to_numpy())
                    self.future.append(op_sample.drop(columns=['Timestamp']).to_numpy())
                    

                    # Move to the next batch of data in IN
                    start_row += in_step
                    sample_index += 1

        return self.past, self.future

    def getfeature(self, featurepath):
        """
        Load the batch of images (or feature vectors) from the given path.
        """
        data_batch = []
        try:
            # Load each .npy file in featurepath
            for file_path in featurepath:
                data = np.load(file_path)
                data_batch.append(data)
        except FileNotFoundError:
            print(f"Error: The file was not found.")
        except Exception as e:
            print(f"An error has occurred: {e}")

        return np.array(data_batch)

    def custom_collate_fn(self, batch):
        imgs = [item[0]["Image"] for item in batch]
        pcds = [item[0]["PCD"] for item in batch]
        img_feature_path = [item[0]["img_paths"] for item in batch]
        pcd_feature_path = [item[0]["pcd_paths"] for item in batch]
        
        ys = [item[1] for item in batch]  # Speed & Steer
        ttv = [item[2] for item in batch]

        # Convert to np.array once, then to torch.tensor
        imgs_np = np.stack(imgs, axis=0)    # shape -> (batch_size, ..., ...)
        imgs_t  = torch.from_numpy(imgs_np).float()

        pcds_np = np.stack(pcds, axis=0)
        pcds_t  = torch.from_numpy(pcds_np).float()

        # If Speed and Steer are also numpy arrays, do a similar np.stack.
        # Otherwise if they are already torch tensors, you can just stack them in torch.
        speeds = torch.stack([torch.as_tensor(y["Speed"]) for y in ys], dim=0).float()
        steers = torch.stack([torch.as_tensor(y["Steer"]) for y in ys], dim=0).float()

        return {
            "Image": imgs_t,
            "PCD": pcds_t,
            "Speed": speeds,
            "Steer": steers,
            "TTV": ttv,
            "img_paths": img_feature_path,
            "pcd_paths": pcd_feature_path
        }

    def create_train_test_split(self):
        """
        This function splits one dataset object into two subsets:
        - train_dataset (where label == 1)
        - test_dataset (where label == 0)
        """
        train_indices = []
        test_indices = []

        for idx in range(len(self)):
            # We only need to check the label from the first row for the sample
            # item is (x, y, label)
            _, _, label = self[idx]
            if label == 1:
                train_indices.append(idx)
            else:
                test_indices.append(idx)

        train_subset = Subset(self, train_indices)
        test_subset  = Subset(self, test_indices)
        
        return train_subset, test_subset

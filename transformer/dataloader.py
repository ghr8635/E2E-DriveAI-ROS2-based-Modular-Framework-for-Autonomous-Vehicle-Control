import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def dataloader(camera_features_path, lidar_features_path, labels_path):
    # Assuming each feature file corresponds to a timestamp
    camera_features = [np.load(f"{camera_features_path}/{file}") for file in sorted(os.listdir(camera_features_path))]
    lidar_features = [np.load(f"{lidar_features_path}/{file}") for file in sorted(os.listdir(lidar_features_path))]

    # Load labels
    labels_df = pd.read_excel(labels_path)
    steering_angles = labels_df['steering_angle'].values
    velocities = labels_df['velocity'].values

    # Combine features (camera + LiDAR)
    combined_features = [np.concatenate((cam, lidar), axis=1) for cam, lidar in zip(camera_features, lidar_features)]

    # Normalize combined features and labels
    scaler_features = MinMaxScaler()
    scaler_labels = MinMaxScaler()

    combined_features = scaler_features.fit_transform(np.array(combined_features).reshape(-1, combined_features[0].shape[1]))
    labels = scaler_labels.fit_transform(np.column_stack((steering_angles, velocities)))

    seq_length = 48
    X, y = create_sequences(combined_features, labels, seq_length)

    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    return (X_train, X_val, y_train, y_val)


def create_sequences(features, labels, seq_length=48):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i + seq_length])
        y.append(labels[i + seq_length])
    return np.array(X), np.array(y)

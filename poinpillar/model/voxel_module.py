import torch
import torch.nn as nn

class HardVoxelization(torch.nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super(HardVoxelization, self).__init__()
        # Initialize these tensors without device specification
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        self.point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)
        self.max_num_points = max_num_points

        # Handle max_voxels as a tuple or a single value
        if isinstance(max_voxels, tuple):
            self.max_voxels_train, self.max_voxels_infer = max_voxels
        else:
            self.max_voxels_train = self.max_voxels_infer = max_voxels

        # Default max_voxels for backward compatibility
        self.max_voxels = self.max_voxels_train

        # Compute the grid size for voxels (this must be done in __init__)
        self.grid_size = None  # Placeholder, will be computed in forward

    def forward(self, points, is_training=True):
        device = points.device  # Get the device of the input tensor
        
        # Move point_cloud_range and voxel_size to the same device as the points tensor
        point_cloud_range = self.point_cloud_range.to(device)
        voxel_size = self.voxel_size.to(device)
        
        # Compute grid_size on the fly in the forward pass, ensuring it's on the right device
        self.grid_size = ((point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size).int().to(device)
        
        # Use appropriate max_voxels based on the training/inference context
        max_voxels = self.max_voxels_train if is_training else self.max_voxels_infer
        
        ndim = points.shape[1]

        # Step 1: Compute voxel indices
        voxel_indices = torch.floor((points[:, :3] - point_cloud_range[:3]) / voxel_size).int()
        mask = ((voxel_indices >= 0) & (voxel_indices < self.grid_size)).all(dim=1)
        points = points[mask]
        voxel_indices = voxel_indices[mask]

        # Step 2: Find unique voxel indices
        unique_indices, inverse_indices = torch.unique(voxel_indices, return_inverse=True, dim=0)
        num_voxels = min(unique_indices.shape[0], max_voxels)

        # Step 3: Initialize outputs
        voxels = torch.zeros((num_voxels, self.max_num_points, ndim), dtype=points.dtype, device=device)
        coordinates = torch.zeros((num_voxels, 3), dtype=torch.int32, device=device)
        num_points_per_voxel = torch.zeros((num_voxels,), dtype=torch.int32, device=device)

        # Step 4: Fill voxels
        for i in range(points.shape[0]):
            idx = inverse_indices[i]
            if idx < num_voxels and num_points_per_voxel[idx] < self.max_num_points:
                point_idx = num_points_per_voxel[idx]
                voxels[idx, point_idx] = points[i]
                num_points_per_voxel[idx] += 1

        coordinates[:num_voxels] = unique_indices[:num_voxels]

        return voxels, coordinates, num_points_per_voxel


# Voxelization wrapper class (similar to Voxelization in the original code)
class Voxelization(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super(Voxelization, self).__init__()
        self.voxelizer = HardVoxelization(voxel_size, point_cloud_range, max_num_points, max_voxels)

    def forward(self, points, is_training=True):
        return self.voxelizer(points, is_training)


# Example usage
if __name__ == "__main__":
    # Random example points
    points = torch.rand((10000, 4))  # N points with x, y, z, and 1 feature

    # Voxelization parameters
    voxel_size = [0.2, 0.2, 0.2]
    point_cloud_range = [0, 0, 0, 20, 20, 20]
    max_num_points = 10
    max_voxels = (2000, 4000)  # Training and inference max_voxels

    # Initialize the voxelization module
    voxelizer = Voxelization(voxel_size, point_cloud_range, max_num_points, max_voxels)

    # Perform voxelization
    voxels, coordinates, num_points = voxelizer(points, is_training=True)

    print("Voxels shape:", voxels.shape)
    print("Coordinates shape:", coordinates.shape)
    print("Num points per voxel shape:", num_points.shape)


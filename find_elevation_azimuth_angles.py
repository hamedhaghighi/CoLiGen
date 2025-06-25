import os
import numpy as np
from sklearn.cluster import KMeans
from glob import glob



if __name__ == '__main__':
    # Path to all point cloud files
    # Path to all point cloud files
    point_paths = glob('/home/extraspace/WADS/original_dataset/dataset/sequences/00/velodyne/*.bin')

    # Number of pitch channels
    num_pitch_channels = 64

    # Storage for accumulated sorted pitch centers
    all_pitch_centers = []

    # Iterate over all point clouds
    for point_path in point_paths:
        # Load point cloud
        point_cloud = np.fromfile(point_path, dtype=np.float32).reshape((-1, 4))

        # Extract xyz coordinates
        xyz = point_cloud[:, :3]
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        # Compute depth and pitch
        depth = np.linalg.norm(xyz, axis=1)
        pitch = np.arcsin(z / depth).reshape(-1, 1)  # Vertical angles

        # Apply K-Means clustering to find 64 pitch channels
        kmeans = KMeans(n_clusters=num_pitch_channels, random_state=42, n_init=10)
        kmeans.fit(pitch)

        # Sort cluster centers in ascending order
        sorted_centers = np.sort(kmeans.cluster_centers_.flatten())[::-1]

        # Store sorted centers
        all_pitch_centers.append(sorted_centers)

    # Convert to NumPy array for efficient computation
    all_pitch_centers = np.array(all_pitch_centers)  # Shape: (num_files, 64)

    mean_pitch_per_channel = np.mean(all_pitch_centers, axis=0)
    import pdb; pdb.set_trace()

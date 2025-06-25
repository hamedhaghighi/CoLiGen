"""
Script to find elevation and azimuth angles for LIDAR point cloud processing.

This script analyzes LIDAR point cloud data to determine the optimal elevation
angles for different channels by clustering pitch values across multiple scans.
The resulting angles can be used to configure LIDAR sensor parameters or
for point cloud projection and processing.
"""

# Standard library imports
import os
from glob import glob

# Third-party imports
import numpy as np
from sklearn.cluster import KMeans

if __name__ == "__main__":
    # Path to all point cloud files
    # Path to all point cloud files
    point_paths = glob(
        "/home/extraspace/WADS/original_dataset/dataset/sequences/00/velodyne/*.bin"
    )

    # Number of pitch channels to extract from the LIDAR data
    # This corresponds to the number of vertical channels in the LIDAR sensor
    num_pitch_channels = 64

    # Storage for accumulated sorted pitch centers from all point clouds
    # Each point cloud will contribute 64 pitch angles (one per channel)
    all_pitch_centers = []

    # Iterate over all point cloud files to analyze their pitch distributions
    for point_path in point_paths:
        # Load point cloud data from binary file
        # Each point has 4 values: x, y, z, intensity
        point_cloud = np.fromfile(point_path, dtype=np.float32).reshape((-1, 4))

        # Extract xyz coordinates (first 3 columns)
        # x: forward direction, y: left direction, z: up direction
        xyz = point_cloud[:, :3]
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        # Compute depth (distance from origin) and pitch (elevation angle)
        # depth: Euclidean distance from LIDAR origin to each point
        # pitch: vertical angle in radians (positive = above horizon, negative = below)
        depth = np.linalg.norm(xyz, axis=1)  # Euclidean distance from origin
        pitch = np.arcsin(z / depth).reshape(-1, 1)  # Vertical angles in radians

        # Apply K-Means clustering to find 64 distinct pitch channels
        # This groups similar pitch values together to identify the main elevation angles
        # The clustering helps identify the discrete vertical channels of the LIDAR sensor
        kmeans = KMeans(n_clusters=num_pitch_channels, random_state=42, n_init=10)
        kmeans.fit(pitch)

        # Sort cluster centers in descending order (from highest to lowest elevation)
        # This ensures consistent ordering across different point clouds
        sorted_centers = np.sort(kmeans.cluster_centers_.flatten())[::-1]

        # Store sorted centers for this point cloud
        all_pitch_centers.append(sorted_centers)

    # Convert to NumPy array for efficient computation
    # Shape: (num_files, 64) - each row represents pitch centers from one point cloud
    all_pitch_centers = np.array(all_pitch_centers)

    # Calculate mean pitch angle for each channel across all point clouds
    # This gives us the average elevation angle for each of the 64 channels
    # The result represents the typical pitch angles for this LIDAR sensor configuration
    mean_pitch_per_channel = np.mean(all_pitch_centers, axis=0)

    # Debug breakpoint for inspection of results
    # This allows manual inspection of the computed pitch angles
    import pdb

    pdb.set_trace()

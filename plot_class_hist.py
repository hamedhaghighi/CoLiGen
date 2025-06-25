"""
Script to analyze and plot class distribution histograms for LIDAR datasets.

This script loads LIDAR point cloud data and semantic labels to compute:
1. Class distribution histograms
2. Statistical properties (mean, standard deviation) of point cloud features
3. Depth statistics
4. Point count statistics

The results can be used to understand dataset characteristics and class imbalance.
"""

# Standard library imports
import os.path as osp
from glob import glob

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

# Local imports
from util import _map


def load_datalist(root):
    """
    Load file paths for labels and point clouds from all dataset subsets.

    This function scans through all sequence directories (00-09) and collects
    paths to label files and point cloud files for analysis.

    Args:
        root: Root directory containing the dataset sequences

    Returns:
        tuple: (label_paths_array, point_cloud_paths_array) - Arrays of file paths
    """
    subsets = range(10)  # Sequences 00-09
    datalist_label = []
    datalist_points = []

    # Iterate through each sequence subset
    for subset in subsets:
        subset_dir = osp.join(root, str(subset).zfill(2))

        # Get sorted lists of label and point cloud files
        # This ensures consistent ordering across different runs
        sub_label_path = sorted(glob(osp.join(subset_dir, "labels/*")))
        sub_point_path = sorted(glob(osp.join(subset_dir, "velodyne/*")))

        # Add to master lists
        datalist_label += list(sub_label_path)
        datalist_points += list(sub_point_path)

    return np.array(datalist_label), np.array(datalist_points)


def main():
    """
    Main function to analyze dataset statistics and class distributions.

    This function:
    1. Loads dataset configuration and file paths
    2. Processes point clouds and labels to compute statistics
    3. Calculates class distribution histograms
    4. Prints summary statistics
    5. Optionally plots histograms (currently commented out)
    """
    # Configuration parameters
    is_raw = False  # Whether data is in raw binary format or processed numpy format
    np.random.seed(0)  # Set random seed for reproducibility
    dataset_name = "carla"  # Dataset name for configuration loading

    # Load dataset configuration from YAML file
    # This contains label mappings and dataset-specific parameters
    ds_cfg = yaml.safe_load(open(f"configs/dataset_cfg/{dataset_name}_cfg.yml", "r"))
    data_dir = osp.join(ds_cfg["data_dir"], "sequences")
    label_id_list = list(ds_cfg["learning_map"].keys())  # List of valid label IDs

    # Set up visualization parameters
    cm = plt.get_cmap("gist_rainbow")  # Color map for plotting
    hist = dict()  # Dictionary to store class histogram counts

    # Load file paths and randomly sample for analysis
    label_path_list, point_path_list = load_datalist(data_dir)
    idx_array = np.arange(len(label_path_list))
    np.random.shuffle(idx_array)

    # Limit analysis to 5000 samples for efficiency
    # This provides a good balance between accuracy and computation time
    label_path_list, point_path_list = (
        label_path_list[idx_array][:5000],
        point_path_list[idx_array][:5000],
    )

    # Initialize statistics accumulators
    # mu: mean values for [depth, x, y, z, intensity]
    # std: standard deviation values for [depth, x, y, z, intensity]
    mu, std = np.zeros(5), np.zeros(5)
    n_points = 0  # Total number of points across all samples
    max_depth = 0  # Maximum depth value encountered

    # Process each point cloud and label pair
    for p_l, l_p in tqdm(zip(point_path_list, label_path_list)):
        # Load point cloud data based on format
        if is_raw:
            # Load raw binary format (KITTI format)
            # Each point has 4 values: x, y, z, intensity
            point_cloud = np.fromfile(p_l, dtype=np.float32).reshape((-1, 4))
            # Calculate depth as Euclidean distance from origin
            depth = np.linalg.norm(point_cloud[:, :3], ord=2, axis=1)
        else:
            # Load processed numpy format
            # Data is stored as channels: [depth, x, y, z, intensity]
            point_cloud = np.load(p_l).astype(np.float32)
            depth = point_cloud[0].reshape(-1)  # First channel is depth
            # Extract x, y, z, intensity and transpose to (N, 4) format
            point_cloud = np.transpose(point_cloud[1:5].reshape(4, -1), (1, 0))

        # Accumulate statistics across all samples
        n_points += len(point_cloud)
        max_depth = max(max_depth, np.max(depth))

        # Update mean and variance accumulators using online algorithm
        mu[0] += depth.sum()  # Depth mean accumulator
        mu[1:] += point_cloud.sum(axis=0)  # x, y, z, intensity mean accumulators
        std[0] += (depth**2).sum()  # Depth variance accumulator
        std[1:] += (point_cloud**2).sum(
            axis=0
        )  # x, y, z, intensity variance accumulators

        # Load and process semantic labels
        if is_raw:
            # Load raw label format (KITTI format)
            label_id_array = np.fromfile(l_p, dtype=np.int32)
            label_id_array = label_id_array & 0xFFFF  # Extract lower 16 bits
            # Map to learning labels and back to original for consistency
            label_id_array = _map(label_id_array, ds_cfg["learning_map"])
            label_id_array = _map(label_id_array, ds_cfg["learning_map_inv"])
        else:
            # Load image format labels (PNG files)
            label_id_array = np.array(Image.open(l_p))
            # Map to original label IDs
            label_id_array = _map(label_id_array, ds_cfg["learning_map_inv"])

        # Count points for each class to build histogram
        for id in label_id_list:
            s = (label_id_array == id).sum()  # Count points of this class
            if s > 0:
                if id in hist:
                    hist[id] += s
                else:
                    hist[id] = s

    # Calculate final statistics from accumulators
    mu = mu / n_points  # Convert sums to means
    std = np.sqrt((std / n_points) - mu**2)  # Calculate standard deviations

    # Print summary statistics
    print("max_depth:", max_depth)
    print("mu:", mu)
    print("std:", std)
    # Print normalized histogram (frequencies instead of counts)
    print("hist", {k: v / n_points for k, v in hist.items()})

    # Optional: Plot histogram (currently commented out)
    # This code would create a bar chart showing class distribution
    # # Plot the histogram
    # num_label =len(hist)
    # color_list = [cm(i/num_label) for i in range(num_label)]
    # np.random.shuffle(color_list)
    # classes = np.arange(num_label)
    # for i, (k , v) in enumerate(hist.items()):
    #     bar = plt.bar([i], [v], label=k)
    #     bar[0].set_color(color_list[i])
    # plt.xlabel('Semantic Label')
    # plt.ylabel('Frequency')
    # plt.title(f'Histogram of Semantic Labels in {dataset_name} Dataset')
    # plt.xticks(classes)
    # plt.legend()
    # ax = plt.gca()
    # leg = ax.get_legend()
    # for i, lgh in enumerate(leg.legendHandles):
    #     lgh.set_color(color_list[i])
    # plt.show()


if __name__ == "__main__":
    main()

"""
Script to find nearest neighbor matches between synthetic and real LIDAR data.

This script compares synthetic LIDAR point clouds (e.g., from CARLA) with real
LIDAR data to find the most similar real samples based on depth error metrics.
This is useful for evaluating the quality of synthetic data generation or for
finding corresponding real-world data for synthetic samples.
"""

# Standard library imports
import argparse
import os
import random
import shutil
import time
from collections import defaultdict

# Third-party imports
import numpy as np
import torch
import tqdm
import yaml

# Local imports
from dataset.datahandler import get_data_loader, get_dataset
from dataset.kitti_odometry import KITTIOdometry
from fid import FID
# from data import create_dataset
from models import create_model
from rangenet.tasks.semantic.modules.segmentator import *
from util import *
from util.lidar import LiDAR
from util.metrics.cov_mmd_1nna import compute_cov_mmd_1nna
from util.metrics.depth import compute_depth_error
from util.metrics.jsd import compute_jsd
from util.metrics.seg_accuracy import compute_seg_accuracy
from util.metrics.swd import compute_swd
from util.sampling.fps import downsample_point_clouds
from util.visualizer import Visualizer

# Environment setup for compatibility
# Set LD_PRELOAD to ensure proper C++ library loading
os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def inv_to_xyz(inv, lidar, tol=1e-8):
    """
    Convert inverse depth representation to 3D point cloud coordinates.

    This function takes inverse depth values (from model output) and converts
    them back to 3D Cartesian coordinates using the LiDAR configuration.

    Args:
        inv: Inverse depth tensor from model output
        lidar: LiDAR configuration object containing projection parameters
        tol: Tolerance for numerical stability in coordinate conversion

    Returns:
        torch.Tensor: 3D point cloud coordinates with shape (B, N, 3)
    """
    # Convert from tanh to sigmoid activation and clamp to valid range [0, 1]
    inv = tanh_to_sigmoid(inv).clamp_(0, 1)

    # Convert inverse depth to 3D coordinates using LiDAR projection
    xyz = lidar.inv_to_xyz(inv, tol)

    # Reshape and transpose to (batch_size, num_points, 3) format
    xyz = xyz.flatten(2).transpose(1, 2)  # (B,N,3)

    # Downsample point cloud to 512 points for computational efficiency
    xyz = downsample_point_clouds(xyz, 512)
    return xyz


def main(runner_cfg_path=None):
    """
    Main function to find nearest neighbor matches between synthetic and real data.

    This function:
    1. Loads synthetic and real datasets
    2. For each synthetic sample, finds the most similar real sample
    3. Uses depth error as the similarity metric
    4. Reports the best matches found

    Args:
        runner_cfg_path: Optional path to runner configuration file
    """
    # Configuration for reference dataset
    ref_dataset_name = "semanticPOSS"
    split = "train/val"

    # Define specific sequences and frame IDs to analyze
    # These represent specific synthetic samples we want to find matches for
    if ref_dataset_name == "semanticPOSS":
        seqs = [0, 0, 5]  # Sequence numbers
        ids = [75, 385, 200]  # Frame IDs within sequences
    else:
        seqs = [0, 0, 2, 5]
        ids = [1, 268, 345, 586]

    # seqs = [0, 0]
    # ids = [0, 1]

    # Set random seeds for reproducible results
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # DATA = yaml.safe_load(open(pa.cfg_dataset, 'r'))
    ## test whole code fast

    # Dataset configuration
    ds_synth_name = "carla"  # Synthetic dataset name (e.g., CARLA)
    ds_real_name = "semanticPOSS"  # Real dataset name (e.g., SemanticPOSS)
    gpu_id = 0
    device = torch.device("cuda:{}".format(gpu_id))

    # Load dataset configurations from YAML files
    ds_cfg_A = make_class_from_dict(
        yaml.safe_load(open(f"configs/dataset_cfg/{ds_synth_name}_cfg.yml", "r"))
    )
    ds_cfg_B = make_class_from_dict(
        yaml.safe_load(open(f"configs/dataset_cfg/{ds_real_name}_cfg.yml", "r"))
    )

    # LiDAR configuration parameters for both datasets
    # These define the resolution of the depth maps
    width, height = 64, 256
    lidar_A = LiDAR(cfg=ds_cfg_A, height=height, width=width).to(device)
    lidar_B = LiDAR(cfg=ds_cfg_B, height=height, width=width).to(device)

    # Dataset directories
    ds_synth_dir = ds_cfg_A.data_dir
    ds_real_dir = ds_cfg_B.data_dir

    # Initialize synthetic dataset (e.g., CARLA)
    # This loads the synthetic LIDAR data with depth, reflectance, and labels
    sim_dataset = KITTIOdometry(
        ds_synth_dir,
        split,
        ds_cfg_A,
        shape=(height, width),
        flip=False,
        modality=["depth", "reflecance", "label"],
        fill_in_label=True,
        name=ds_synth_name,
        limited_view=False,
        finesize=None,
        norm_label=False,
        is_ref_semposs=False,
    )

    # Initialize real dataset (e.g., SemanticPOSS)
    # This loads the real LIDAR data with the same modalities
    real_dataset = KITTIOdometry(
        ds_real_dir,
        split,
        ds_cfg_B,
        shape=(height, width),
        flip=False,
        modality=["depth", "reflecance", "label"],
        fill_in_label=True,
        name=ds_real_name,
        limited_view=False,
        finesize=None,
        norm_label=False,
        is_ref_semposs=False,
    )

    # Get data list and find indices of selected synthetic samples
    data_list = sim_dataset.datalist
    dataset_A_datalist = np.array(data_list)
    dataset_A_selected_idx = []
    n_sub_sample = min(
        len(real_dataset), 5000
    )  # Number of real samples to compare against

    # Find indices of the specified synthetic samples in the dataset
    for seq, id in zip(seqs, ids):
        # Construct the file path for the synthetic point cloud
        pcl_file_path = os.path.join(
            ds_cfg_A.data_dir,
            "sequences",
            str(seq).zfill(2),
            "velodyne",
            str(id).zfill(6) + (".bin" if ds_cfg_A.is_raw else ".npy"),
        )
        # Find the index of this file in the dataset
        dataset_A_selected_idx.append(
            np.where(dataset_A_datalist == pcl_file_path)[0][0]
        )

    # Progress bar for synthetic samples
    val_tq = tqdm.tqdm(total=len(dataset_A_selected_idx), desc="sim_Iter", position=5)

    # Iterate through each selected synthetic sample
    for i, idx in enumerate(dataset_A_selected_idx):
        # Load synthetic data and prepare for processing
        sim_data = sim_dataset[idx]
        sim_data = {
            k: v.unsqueeze(0) for k, v in sim_data.items() if not isinstance(v, str)
        }
        sim_data = fetch_reals(sim_data, lidar_A, device, False)

        # Progress bar for real samples comparison
        real_tq = tqdm.tqdm(total=n_sub_sample, desc="real_Iter", position=5)
        min_rmse = np.inf  # Track minimum RMSE error
        min_path = None  # Track path of best matching real sample

        # Randomly sample real data for comparison to reduce computation time
        sub_real_d_indices = np.random.choice(
            len(real_dataset), n_sub_sample, replace=False
        )

        # Compare synthetic sample with each real sample
        for jdx in sub_real_d_indices:
            # Load real data and prepare for processing
            real_data = real_dataset[jdx]
            real_data_path = real_data["path"]
            real_data = {
                k: v.unsqueeze(0)
                for k, v in real_data.items()
                if not isinstance(v, str)
            }
            real_data = fetch_reals(real_data, lidar_B, device, False)

            # Compute depth error between synthetic and real data
            # RMSE (Root Mean Square Error) is used as similarity metric
            curr_rmse = compute_depth_error(sim_data["depth"], real_data["depth"])[
                "rmse"
            ]

            # Update best match if current error is lower
            if curr_rmse < min_rmse:
                min_rmse = curr_rmse
                min_path = real_data_path
            real_tq.update(1)

        # Print the best match found for this synthetic sample
        print("sim seq, id:", seqs[i], ids[i], "=>", min_path)
        val_tq.update(1)


if __name__ == "__main__":
    main()

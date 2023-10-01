import time
# from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from fid import FID
from dataset.datahandler import get_data_loader, get_dataset
from dataset.kitti_odometry import KITTIOdometry
from rangenet.tasks.semantic.modules.segmentator import *
import yaml
import argparse
import numpy as np
import torch
import tqdm
import os
from util.lidar import LiDAR
from util import *
from collections import defaultdict
import shutil
from util.sampling.fps import downsample_point_clouds
from util.metrics.cov_mmd_1nna import compute_cov_mmd_1nna
from util.metrics.jsd import compute_jsd
from util.metrics.swd import compute_swd
from util.metrics.seg_accuracy import compute_seg_accuracy
from util.metrics.depth import compute_depth_error
import random


os.environ['LD_PRELOAD'] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def inv_to_xyz(inv, lidar, tol=1e-8):
    inv = tanh_to_sigmoid(inv).clamp_(0, 1)
    xyz = lidar.inv_to_xyz(inv, tol)
    xyz = xyz.flatten(2).transpose(1, 2)  # (B,N,3)
    xyz = downsample_point_clouds(xyz, 512)
    return xyz




def main(runner_cfg_path=None):
    
    ref_dataset_name = 'kitti'
    split = 'train/val'
    # if ref_dataset_name == 'semanticPOSS':
    #     seqs = [0, 0, 5] 
    #     ids = [75, 385, 200]
    # else:
    #     seqs = [0, 0,  2, 5]
    #     ids = [1, 268, 345, 586]

    seqs = [0, 0]
    ids = [0, 1]
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
        
    # DATA = yaml.safe_load(open(pa.cfg_dataset, 'r'))
    ## test whole code fast

    ds_synth_name = 'carla'
    ds_real_name = 'kitti'
    gpu_id = 0
    device = torch.device('cuda:{}'.format(gpu_id))
    ds_cfg_A = make_class_from_dict(yaml.safe_load(open(f'configs/dataset_cfg/{ds_synth_name}_cfg.yml', 'r')))
    ds_cfg_B = make_class_from_dict(yaml.safe_load(open(f'configs/dataset_cfg/{ds_real_name}_cfg.yml', 'r')))
    width , height = 64, 256
    lidar_A = LiDAR(
    cfg=ds_cfg_A,
    height=height,
    width=width).to(device)
    lidar_B = LiDAR(
    cfg=ds_cfg_B,
    height=height,
    width=width
   ).to(device)

    ds_synth_dir = ds_cfg_A.data_dir
    ds_real_dir = ds_cfg_B.data_dir

    sim_dataset = KITTIOdometry(
          ds_synth_dir,
          split,
          ds_cfg_A,
          shape=(height, width),
          flip=False,
          modality=['depth', 'reflecance', 'label'],
          fill_in_label=True,
          name=ds_synth_name,
          limited_view=False,
          finesize=None,
          norm_label=False,
          is_ref_semposs=False
      )
    real_dataset = KITTIOdometry(
          ds_real_dir,
          split,
          ds_cfg_B,
          shape=(height, width),
          flip=False,
          modality=['depth', 'reflecance', 'label'],
          fill_in_label=True,
          name=ds_real_name,
          limited_view=False,
          finesize=None,
          norm_label=False,
          is_ref_semposs=False
      )

    data_list = sim_dataset.datalist
    dataset_A_datalist = np.array(data_list)
    dataset_A_selected_idx = []
    n_sub_sample = 100
    for seq, id in zip(seqs, ids):
        pcl_file_path = os.path.join(ds_cfg_A.data_dir, 'sequences', str(seq).zfill(2), 'velodyne', str(id).zfill(6)+('.bin' if ds_cfg_A.is_raw else '.npy'))
        dataset_A_selected_idx.append(np.where(dataset_A_datalist == pcl_file_path)[0][0])
    val_tq = tqdm.tqdm(total=len(dataset_A_selected_idx), desc='sim_Iter', position=5)
    for i, idx in enumerate(dataset_A_selected_idx):
        sim_data = sim_dataset[idx]
        sim_data = {k: v.unsqueeze(0) for k, v in sim_data.items() if not isinstance(v, str)}
        sim_data = fetch_reals(sim_data, lidar_A, device, False)
        real_tq = tqdm.tqdm(total=n_sub_sample, desc='real_Iter', position=5)
        min_rmse = np.inf
        min_path = None
        sub_real_d_indices = np.random.choice(len(real_dataset), n_sub_sample, replace=False)
        for jdx in sub_real_d_indices:
            real_data = real_dataset[jdx]
            real_data_path = real_data['path']
            real_data = {k: v.unsqueeze(0) for k, v in real_data.items() if not isinstance(v, str)}   
            real_data = fetch_reals(real_data, lidar_B, device, False)
            curr_rmse = compute_depth_error(sim_data['depth'], real_data['depth'])['rmse']
            if curr_rmse < min_rmse:
                min_rmse = curr_rmse
                min_path = real_data_path
            real_tq.update(1)
        print('sim seq, id:', seqs[i], ids[i], '=>', min_path)
        val_tq.update(1)


if __name__ == '__main__':
    main()
    
 
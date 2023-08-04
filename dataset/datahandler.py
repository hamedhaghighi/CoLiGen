import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
import torch.nn.functional as F
from glob import glob
from util.lidar import point_cloud_to_xyz_image
from util import _map
from dataset.kitti_odometry import KITTIOdometry
from dataset.nuscene import NuScene
import yaml
from util import make_class_from_dict

class BinaryScan(Dataset):

  def __init__(self, dataset_A, dataset_B):
    # save deats
    self.sizeA = len(dataset_A)
    self.sizeB = len(dataset_B)
    self.datasetA, self.datasetB = dataset_A, dataset_B

  def __getitem__(self, index):
    index_A = index % self.sizeA
    index_B = np.random.randint(0, self.sizeB)
    return {'A': self.datasetA[index_A], 'B': self.datasetB[index_B]}

  def __len__(self):
    return max(self.sizeA, self.sizeB)

  @staticmethod
  def map(label, mapdict):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
      if isinstance(data, list):
        nel = len(data)
      else:
        nel = 1
      if key > maxkey:
        maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
      lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
      lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
      try:
        lut[key] = data
      except IndexError:
        print("Wrong key ", key)
    # do the mapping
    return lut[label]

def get_dataset(dataset_name, cfg, ds_cfg, data_dir, split, limited_view=False):
  if dataset_name in ['kitti', 'carla', 'synthlidar', 'semanticPOSS']:
    dataset = KITTIOdometry(
          data_dir,
          split,
          ds_cfg,
          shape=(cfg.img_prop.height, cfg.img_prop.width),
          flip=False,
          modality=cfg.modality,
          fill_in_label=cfg.fill_in_label,
          name=dataset_name,
          limited_view=limited_view,
          finesize=cfg.img_prop.finesize if (split == 'train' and cfg.img_prop.finesize != -1) else None
      )
  elif dataset_name =='nuscene':
    dataset = NuScene(
          data_dir,
          split,
          ds_cfg,
          shape=(cfg.img_prop.height, cfg.img_prop.width),
          flip=False,
          modality=cfg.modality,
          is_sorted=False,
          is_raw=ds_cfg.is_raw,
          fill_in_label=cfg.fill_in_label
      )
  return dataset

def get_data_loader(cfg, split, batch_size, dataset_name='', shuffle=True, two_dataset_enabled=True):
  cfg_A = cfg.dataset.dataset_A
  dataset_name_A = cfg_A.name if dataset_name == '' else dataset_name
  ds_cfg_A = make_class_from_dict(yaml.safe_load(open(f'configs/dataset_cfg/{dataset_name_A}_cfg.yml', 'r')))
  data_dir = cfg_A.data_dir if dataset_name == '' else ds_cfg_A.data_dir
  limited_view = 'rgb' in cfg.model.modality_A or 'rgb' in cfg.model.modality_B
  dataset_A = get_dataset(dataset_name_A, cfg_A, ds_cfg_A, data_dir, split, limited_view)
  dataset = dataset_A
  if hasattr(cfg.dataset, 'dataset_B') and two_dataset_enabled:
    cfg_B = cfg.dataset.dataset_B
    ds_cfg_B = make_class_from_dict(yaml.safe_load(open(f'configs/dataset_cfg/{cfg_B.name}_cfg.yml', 'r')))
    dataset_B = get_dataset(cfg.dataset.dataset_B.name, cfg_B, ds_cfg_B, cfg_B.data_dir, split, limited_view)
    dataset = BinaryScan(dataset_A, dataset_B)
  loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, drop_last=True if split == 'train' else False)
  return loader, dataset








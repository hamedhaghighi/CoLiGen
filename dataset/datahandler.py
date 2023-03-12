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


class BinaryScan(Dataset):

  def __init__(self, data_dirA, data_statsA, data_dirB, data_statsB, max_dataset_size=-1):
    # save deats
    self.sizeA = len(data_statsA)
    self.sizeB = len(data_statsB)
    
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



def get_data_loader(cfg, split, batch_size, dataset_name='kitti'):
  cfg = cfg.dataset_A
  if dataset_name == 'kitti':
    dataset = KITTIOdometry(
          cfg.data_dir,
          split,
          cfg,
          shape=(cfg.img_prop.height, cfg.img_prop.width),
          flip=False,
          modality=cfg.modality,
          is_sorted=cfg.is_sorted,
          is_raw=cfg.is_raw,
          fill_in_label=cfg.fill_in_label
      )
  elif dataset_name =='nuscene':
    dataset = NuScene(
          cfg.data_dir,
          split,
          cfg,
          shape=(cfg.img_prop.height, cfg.img_prop.width),
          flip=False,
          modality=cfg.modality,
          is_sorted=False,
          is_raw=cfg.is_raw,
          fill_in_label=cfg.fill_in_label
      )
  loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,
                                                    shuffle= (split == 'train'),
                                                    num_workers=4)
  return loader, dataset








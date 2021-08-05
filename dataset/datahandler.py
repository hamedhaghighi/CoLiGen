import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
from .laserscan import LaserScan, SemLaserScan
import torch.nn.functional as F
from torchvision import transforms
import yaml
import cv2
import glob



def normalize(array):
  if len(array.shape) > 1:
    min = array.min(axis=0) if len(array.shape) > 1 else array.min()
    max = array.max(axis=0) if len(array.shape) > 1 else array.max()
    array = (array - min)/(max - min)
    return array

class UnaryScan(Dataset):

  def __init__(self, data_dir, data_stats, max_dataset_size=-1):

    self.is_labeled = data_stats['have_rgb']
    self.color_included = data_stats['have_label']
    cfg = yaml.safe_load(open('../configs/semantic-kitti.yaml', 'r'))
    self.proj_H = cfg['sensor']['img_prop']['height'],
    self.proj_W = cfg['sensor']['img_prop']['width'],
    self.fov_up = cfg['sensor']['fov_up'],
    self.fov_down = cfg['sensor']['fov_down'],
    self.foh_left = cfg['sensor']['foh_left'],
    self.foh_right = cfg['sensor']['foh_right']

    self.data_dir = data_dir
    self.scan_file_names = glob.glob(data_dir + 'sequences/*/velodyne/*')
    self.scan_file_names.sort()
    rand_order = np.random.permutation(np.arange(len(self.scan_file_names)))
    self.scan_file_names = [self.scan_file_names[ind] for ind in rand_order]
    if max_dataset_size!=-1:
      self.scan_file_names = self.scan_file_names[:max_dataset_size]
    
    if self.is_labeled:
      self.label_filenames = glob.glob(data_dir + 'sequences/*/labels/*')
      self.label_filenames.sort()
      self.label_filenames = [self.label_filenames[ind] for ind in rand_order]
      if max_dataset_size != -1:
        self.label_filenames = self.label_filenames[:max_dataset_size]

    if os.path.isdir(self.data_dir):
      print("Sequences folder exists! Using sequences from %s" % self.data_dir)
    else:
      raise ValueError("Sequences folder doesn't exist! Exiting...")
    

  

  def __getitem__(self, index):
    # get item in tensor shape
    scan_file = self.scan_file_names[index]
    label_file =self.label_filenames[index]
    scan = np.fromfile(scan_file, dtype=np.float32)
    channels = 7 if self.color_included else 4
    scan = scan.reshape((-1, channels))
    # put in attribute
    points = scan[:, 0:3]    # get xyz
    remissions = scan[:, 3]  # get remission
    points_rgb = scan[:, 4:].astype('uint8') if self.color_included else None
    if self.is_labeled:
      label = np.fromfile(label_file, dtype=np.int32)
      label = label.reshape((-1))
      sem_label = label & 0xFFFF  # semantic label in lower half
      # inst_label = label >> 16    # instance id in upper half
      assert len(label) == points.shape[0]
    proj_range = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
    # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
    proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1, dtype=np.float32)
    proj_remission = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
    proj_points_rgb = np.full((self.proj_H, self.proj_W, 3), 0, dtype=np.uint8) if self.color_included else None
    # projected index (for each pixel, what I am in the pointcloud)
    # [H,W] index (-1 is no data)
    proj_idx = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)
    proj_sem_label = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)
    # projected remission - [H,W] intensity (-1 is no data)
                            

    # mask containing for each pixel, if it contains a point or not
    proj_mask = np.ones((self.proj_H, self.proj_W), dtype=np.int32)       # [H,W] mask
    fov_up = self.proj_fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad
    foh_left = self.proj_foh_left / 180.0 * np.pi
    foh_right = self.proj_foh_right / 180.0 * np.pi
    foh = abs(foh_left) + abs(foh_right)

    # get depth of all points
    depth = np.linalg.norm(points, 2, axis=1)

    # get scan components
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    # proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_x = (yaw + foh_left)/foh
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]
    mask_x = np.logical_and(proj_x >= 0, proj_x <= 1)
    mask_y = np.logical_and(proj_y >= 0, proj_y <= 1)
    mask = np.logical_and(mask_x, mask_y)

    proj_x = proj_x[mask] * self.proj_W
    proj_y = proj_y[mask] * self.proj_H
    proj_x = proj_x.astype(np.int32)  # store a copy in orig order
    proj_y = proj_y.astype(np.int32)  # stope a copy in original order

    points = points[mask]
    remissions = remissions[mask]
    points_rgb = points_rgb[mask] if points_rgb is not None else None
    sem_label = sem_label[mask] if self.is_labeled else None
    depth = depth[mask]

    # copy of depth in original order
    unproj_range = np.copy(depth)
    # order in decreasing depth
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]
    points = points[order]
    remission = remissions[order]
    sem_label = sem_label[order]
    points_rgb = points_rgb[order] if points_rgb is not None else None
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    # assing to images
    proj_range[proj_y, proj_x] = normalize(depth)
    proj_xyz[proj_y, proj_x] = normalize(points)
    proj_remission[proj_y, proj_x] = normalize(remission)
    proj_points_rgb[proj_y, proj_x] = points_rgb/127.5 -1 if points_rgb is not None else None
    proj_sem_label[proj_y, proj_x] = sem_label if sem_label is not None else None
    proj_idx[proj_y, proj_x] = indices
    proj_mask = (proj_idx >= 0).astype(np.int32)

    proj_mask = torch.from_numpy(proj_mask.repeat(4, axis = 1))
    proj_xyz = torch.from_numpy(proj_xyz.repeat(4, axis=1)) * proj_mask
    proj_range = torch.from_numpy(proj_range.repeat(4, axis=1)) * proj_mask
    proj_remission = torch.from_numpy(proj_remission.repeat(4, axis = 1)) * proj_mask
    proj_rgb = torch.from_numpy(proj_points_rgb.repeat(4, axis=1)) * proj_mask if proj_points_rgb is not None else []
    proj_label = torch.from_numpy(proj_sem_label.repeat(4, axis=1)) * proj_mask if proj_sem_label is not None else []
    return proj_xyz, proj_range, proj_remission, proj_mask, proj_rgb, proj_label

  def __len__(self):
    return len(self.scan_file_names)

class BinaryScan(Dataset):

  def __init__(self, data_dirA, data_statsA, data_dirB, data_statsB, max_dataset_size=-1):
    # save deats
    self.datasetA = UnaryScan(data_dirA, data_statsA, max_dataset_size)
    self.datasetB = UnaryScan(data_dirB, data_statsB, max_dataset_size)
    # get number of classes (can't be len(self.learning_map) because there
    # are multiple repeated entries, so the number that matters is how many
    # there are for the xentropy)
    # sanity checks
    # make sure directory exists 
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


class Loader():
  # standard conv, BN, relu
  def __init__(self,
               data_dict,              # directory for data
               batch_size,        # batch size for train and val
               val_split_ratio,
               workers=4,           # threads to load data
               gt=True,           # get gt?
               shuffle_train=True,
               max_dataset_size=-1, is_train=True, is_training_data=True):  # shuffle training set?

    # number of classes that matters is the one for xentropy
    
    if len(data_dict.keys()) == 2:
      data_dirA, data_dirB = data_dict['dataset_A']['data_dir'], data_dict['dataset_B']['data_dir']
      data_statsA, data_statsB = data_dict['dataset_A']['sensor'], data_dict['dataset_B']['sensor']
      total_dataset = BinaryScan(data_dirA, data_statsA, data_dirB, data_statsB, max_dataset_size)
    else:
      total_dataset = UnaryScan(data_dict['dataset_A']['data_dir'], data_dict['dataset_A']['sensor'], max_dataset_size)

    self.total_dataset = total_dataset

    total_samples = len(total_dataset)

    if is_train:
      assert is_training_data
      train_indcs = range(total_samples)[int(val_split_ratio*total_samples):]
      val_indcs = range(total_samples)[:int(val_split_ratio*total_samples)]
      train_dataset = Subset(total_dataset, train_indcs)
      val_dataset = Subset(total_dataset, val_indcs)
      self.trainloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=shuffle_train,
                                                    num_workers=workers,
                                                    drop_last=True)
      assert len(self.trainloader) > 0

      self.validloader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=workers,
                                                    drop_last=False)
      assert len(self.validloader) > 0

    else:

      if is_training_data:
        val_indcs = range(total_samples)[:int(val_split_ratio*total_samples)]
        test_dataset = Subset(total_dataset, val_indcs)
      else:
        test_dataset = total_dataset 

      self.testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     num_workers=workers,
                                                     drop_last=False)
      assert len(self.testloader) > 0









import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from glob import glob
import random
from util.lidar import point_cloud_to_xyz_image
from util import _map
from PIL import Image
from scipy import ndimage as nd
from nuscenes.nuscenes import NuScenes
import pathlib


MIN_DEPTH = 0.9
MAX_DEPTH = 105.0

class NuScene(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        split,
        DATA,
        shape=(64, 256),
        flip=False,
        modality=("depth"),
        is_sorted=True,
        is_raw=True,
        fill_in_label=False):

      super().__init__()
      self.root = root if is_raw else osp.join(root, 'samples', 'LIDAR_TOP')
      self.split = split
      self.shape = tuple(shape)
      self.min_depth = MIN_DEPTH
      self.max_depth = MAX_DEPTH
      self.flip = flip
      assert "depth" in modality, '"depth" is required'
      self.modality = modality
      self.return_remission = 'reflectance' in self.modality
      self.datalist = None
      self.is_sorted = is_sorted
      self.is_raw = is_raw
      self.DATA = DATA
      self.fill_in_label = fill_in_label
      self.nusc = NuScenes(version = 'v1.0-mini', dataroot = root) if is_raw else None
      self.load_datalist()

    def fill(self, data, invalid=None):
      if invalid is None: invalid = np.isnan(data)
      ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
      return data[tuple(ind)]

    def load_datalist(self):
        if self.is_raw:
            datalist = []
            labels_list=[]
            for i in range(len(self.nusc.sample)):
                sample = self.nusc.sample[i]
                sample_data_token = sample['data']['LIDAR_TOP']
                sample_path = self.nusc.get_sample_data_path(sample_data_token)
                label_path = (pathlib.Path(self.nusc.dataroot) / self.nusc.get("lidarseg", sample_data_token)["filename"])
                datalist.append(sample_path)
                labels_list.append(label_path)
        else:
            datalist = list(sorted(glob(osp.join(self.root, "PCL/*.npy"))))
            labels_list = list(sorted(glob(osp.join(self.root, "label/*.png"))))
        if self.split == 'train':
            split_idx = range(int(len(datalist) * 0.9))
        else:
            split_idx = range(int(len(datalist) * 0.9), len(datalist))
        self.datalist = [datalist[i] for i in split_idx]
        self.labels_list = [labels_list[i] for i in split_idx]

    def preprocess(self, out):
        out["depth"] = np.linalg.norm(out["points"], ord=2, axis=2)
        if 'label' in out and self.fill_in_label:
          fill_in_mask = ~ (out["depth"] > 0.0)
          out['label'] = self.fill(out['label'], fill_in_mask)
        mask = (
            (out["depth"] > 0.0)
            & (out["depth"] > self.min_depth)
            & (out["depth"] < self.max_depth)
        )
        out["depth"] -= self.min_depth
        out["depth"] /= self.max_depth - self.min_depth
        out["mask"] = mask
        out["points"] /= self.max_depth  # unit space
        for key in out.keys():
          if key == 'label' and self.fill_in_label:
            continue
          out[key][~mask] = 0
        return out

    def transform(self, out):
        flip = self.flip and random.random() > 0.5
        for k, v in out.items():
            v = TF.to_tensor(v)
            if flip:
                v = TF.hflip(v)
            v = TF.resize(v, self.shape, TF.InterpolationMode.NEAREST)
            out[k] = v
        return out

    def __getitem__(self, index):
        points_path = self.datalist[index]
        labels_path = self.labels_list[index]
        if not self.is_raw:
            points = np.load(points_path).astype(np.float32)
            sem_label = np.array(Image.open(labels_path))
            points = np.concatenate([points, sem_label.astype('float32')[..., None]], axis=-1)
        else:
            point_cloud = np.fromfile(points_path, dtype=np.float32).reshape((-1, 5))
            point_cloud = point_cloud[:, [1, 0, 2, 3]]; point_cloud[:, 0] = -point_cloud[:, 0]
            sem_label = np.fromfile(labels_path, dtype=np.uint8)
            points, _ = point_cloud_to_xyz_image(np.concatenate([point_cloud, sem_label.astype('float32')[:, None]], axis=1) \
              , H=32, W=1024,fov_up=10.0, fov_down=-30.0, is_sorted=self.is_sorted)
        out = {}
        out["points"] = points[..., :3]
        if "reflectance" in self.modality:
            out["reflectance"] = points[..., [3]] / 255.0
        if "label" in self.modality:
            out["label"] = points[..., [4]]
        out = self.preprocess(out)
        out = self.transform(out)
        return out

    def __len__(self):
        return len(self.datalist)

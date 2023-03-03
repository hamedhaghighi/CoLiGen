import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml
import cv2
from glob import glob
import random
from util.lidar import point_cloud_to_xyz_image
from util import _map
from PIL import Image
from scipy import ndimage as nd

CONFIG = {
    "split": {
        "train": [0, 1, 2, 3, 4, 5, 6, 7, 9],
        "val": [8],
        "test": [10],
        "custom_carla": [0]
    },
}


class KITTIOdometry(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        split,
        DATA,
        shape=(64, 256),
        min_depth=0.9,
        max_depth=120.0,
        flip=False,
        config=CONFIG,
        modality=("depth"),
        is_sorted=True,
        is_raw=True,
        fill_in_label=False):

      super().__init__()
      self.root = osp.join(root, "sequences")
      self.split = split
      self.config = config
      self.subsets = np.asarray(self.config["split"][split])
      self.shape = tuple(shape)
      self.min_depth = min_depth
      self.max_depth = max_depth
      self.flip = flip
      assert "depth" in modality, '"depth" is required'
      self.modality = modality
      self.return_remission = 'reflectance' in self.modality
      self.datalist = None
      self.is_sorted = is_sorted
      self.is_raw = is_raw
      self.DATA = DATA
      self.fill_in_label = fill_in_label
      print(os.getcwd())
      self.load_datalist()


    def fill(self, data, invalid=None):
      if invalid is None: invalid = np.isnan(data)
      ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
      return data[tuple(ind)]

    def load_datalist(self):
        datalist = []
        labels_list=[]
        for subset in self.subsets:
            subset_dir = osp.join(self.root, str(subset).zfill(2))
            sub_point_paths = sorted(glob(osp.join(subset_dir, "velodyne/*")))
            sub_labels_paths = sorted(glob(osp.join(subset_dir, "labels/*")))
            datalist += list(sub_point_paths)
            labels_list += list(sub_labels_paths)
        self.datalist = datalist
        self.labels_list = labels_list

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
            sem_label = _map(sem_label, self.DATA.m_learning_map)
            points = np.concatenate([points, sem_label.astype('float32')[..., None]], axis=-1)
        else:
            point_cloud = np.fromfile(points_path, dtype=np.float32).reshape((-1, 4))
            label = np.fromfile(labels_path, dtype=np.int32)
            sem_label = label & 0xFFFF 
            sem_label = _map(_map(sem_label, self.DATA.learning_map), self.DATA.m_learning_map)
            points, _ = point_cloud_to_xyz_image(np.concatenate([point_cloud, sem_label.astype('float32')[:, None]], axis=1) \
              , H=self.shape[0], W=2048, is_sorted=self.is_sorted)
        out = {}
        out["points"] = points[..., :3]
        if "reflectance" in self.modality:
            out["reflectance"] = points[..., [3]]
        if "label" in self.modality:
            out["label"] = points[..., [4]]
        out = self.preprocess(out)
        out = self.transform(out)
        return out

    def __len__(self):
        return len(self.datalist)

    # def __repr__(self) -> str:
    #     head = "Dataset " + self.__class__.__name__
    #     body = ["Number of datapoints: {}".format(self.__len__())]
    #     body.append("Root location: {}".format(self.root))
    #     lines = [head] + ["    " + line for line in body]
    #     return "\n".join(lines)

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



def get_data_loader(cfg, split, batch_size):
  assert getattr(cfg,'dataset_A')
  cfg = cfg.dataset_A
  dataset = KITTIOdometry(
        cfg.data_dir,
        split,
        cfg,
        shape=(cfg.img_prop.height, cfg.img_prop.width),
        min_depth=cfg.min_depth,
        max_depth=cfg.max_depth,
        flip=False,
        config=CONFIG,
        modality=cfg.modality,
        is_sorted=cfg.is_sorted,
        is_raw=cfg.is_raw,
        fill_in_label=cfg.fill_in_label
    )
  loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,
                                                    shuffle= (split == 'train'),
                                                    num_workers=4)
  return loader, dataset








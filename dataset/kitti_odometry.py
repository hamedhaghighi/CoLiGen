
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
import pykitti

CONFIG = {
    "split": {
        "train": [0, 1, 2, 3, 4, 5, 6, 7, 9],
        "val": [8],
        "test": [10],
        "custom_carla": [0]
    },
}

MIN_DEPTH = 0.9
MAX_DEPTH = 120.0


def car2hom(pc):
    return np.concatenate([pc[:, :3], np.ones((pc.shape[0], 1), dtype=pc.dtype)], axis=-1)

class  KITTIOdometry(torch.utils.data.Dataset):
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
        self.root = osp.join(root, "sequences")
        self.split = split
        self.config = CONFIG
        self.subsets = np.asarray(self.config["split"][split])
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
        if 'rgb' in modality:
            pykitti_dataset = pykitti.odometry(root, '00')
            self.velo_to_camera_rect = pykitti_dataset.calib.T_cam2_velo
            self.cam_intrinsic = pykitti_dataset.calib.P_rect_20
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
        if 'rgb' in self.modality:
            self.rgb_list = datalist.replace('velodyne', 'image_2').replace('bin', 'jpeg')

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
          if (key == 'label' and self.fill_in_label) or (key == 'rgb'):
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

    def image_to_pcl(self, rgb_image, point_cloud):
        rgb = np.zeros((len(point_cloud),3), dtype=np.int32)
        height, width, _ = rgb_image.shape
        hom_pcl_points = car2hom(point_cloud[:, :3]).T
        pcl_in_cam_rect = np.dot(self.velo_to_camera_rect, hom_pcl_points)
        pcl_in_image = np.dot(self.cam_intrinsic, pcl_in_cam_rect)
        pcl_in_image = np.array([pcl_in_image[0] / pcl_in_image[2], pcl_in_image[1] / pcl_in_image[2], pcl_in_image[2]])
        canvas_mask = (pcl_in_image[0] > 0.0) & (pcl_in_image[0] < width) & (pcl_in_image[1] > 0.0)\
            & (pcl_in_image[1] < height) & (pcl_in_image[2] > 0.0)
        valid_pcl_in_image = pcl_in_image[:, canvas_mask].astype('int32')
        rgb[canvas_mask] = rgb_image[valid_pcl_in_image[1], valid_pcl_in_image[0], :]
        return rgb

    def __getitem__(self, index):
        points_path = self.datalist[index]
        labels_path = self.labels_list[index]
        if not self.is_raw:
            points = np.load(points_path).astype(np.float32)
            if "label" in self.modality:
                sem_label = np.array(Image.open(labels_path))
                sem_label = _map(sem_label, self.DATA.m_learning_map)
                points = np.concatenate([points, sem_label.astype('float32')[..., None]], axis=-1)
        else:
            point_cloud = np.fromfile(points_path, dtype=np.float32).reshape((-1, 4))
            if "label" in self.modality:
                label = np.fromfile(labels_path, dtype=np.int32)
                sem_label = label & 0xFFFF 
                sem_label = _map(_map(sem_label, self.DATA.learning_map), self.DATA.m_learning_map)
                point_cloud = np.concatenate([point_cloud, sem_label.astype('float32')[:, None]], axis=1)
            if 'rgb' in self.modality:
                rgb_path = self.rgb_list[index]
                rgb_image = np.array(Image.open(rgb_path))
                rgb = self.image_to_pcl(rgb_image, point_cloud)
                point_cloud = np.concatenate([point_cloud, rgb.astype('float32')], axis=1)
                
            points, _ = point_cloud_to_xyz_image(point_cloud, H=self.shape[0], W=2048, is_sorted=self.is_sorted)
        out = {}
        out["points"] = points[..., :3]
        if "reflectance" in self.modality:
            out["reflectance"] = points[..., [3]]
        if "label" in self.modality:
            out["label"] = points[..., [4]]
        if 'rgb' in self.modality:
            out["rgb"] = points[..., -3:].transpose(2, 0 ,1) / 255.0
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

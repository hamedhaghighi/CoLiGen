import argparse
import multiprocessing
import os
import os.path as osp
from collections import defaultdict
from glob import glob

import joblib
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import yaml
# from datasets.kitti import KITTIOdometry
from util.lidar import point_cloud_to_xyz_image, labelmap

def _map(label, mapdict):
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


# support semantic kitti only for this script

_n_classes = max(labelmap.values()) + 1
_colors = cm.turbo(np.asarray(range(_n_classes)) / (_n_classes - 1))[:, :3] * 255
palette = list(np.uint8(_colors).flatten())




def process_point_clouds(point_path, H=64, W=2048):
    save_dir = lambda x: x.replace("dataset/sequences", "dusty-gan/sequences")
    # setup point clouds
    points = np.fromfile(point_path, dtype=np.float32).reshape((-1, 4))
    # for semantic kitti
    label_path = point_path.replace("/velodyne", "/labels")
    label_path = label_path.replace(".bin", ".label")
    if osp.exists(label_path):
        label = np.fromfile(label_path, dtype=np.int32)
        sem_label = label & 0xFFFF 
        sem_label = _map(sem_label, labelmap)
        points = np.concatenate([points, sem_label.astype('float32')[:, None]], axis=1)
    proj, _ = point_cloud_to_xyz_image(points, H, W, is_sorted=True)


    save_path = save_dir(point_path).replace(".bin", ".npy")
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    np.save(save_path, proj[..., :4])
    if osp.exists(label_path):
        save_path = save_dir(label_path).replace(".label", ".png")
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        labels = Image.fromarray(np.uint8(proj[..., 4]), mode="P")
        labels.putpalette(palette)
        labels.save(save_path)


def mean(tensor, dim):
    tensor = tensor.clone()
    kwargs = {"dim": dim, "keepdim": True}
    valid = (~tensor.isnan()).float()
    tensor[tensor.isnan()] = 0
    tensor = torch.sum(tensor * valid, **kwargs) / valid.sum(**kwargs)
    return tensor


@torch.no_grad()
def compute_avg_angles(loader):

    max_depth = loader.dataset.max_depth

    summary = defaultdict(float)

    for item in tqdm(loader):
        xyz_batch = item["xyz"]

        x = xyz_batch[:, [0]]
        y = xyz_batch[:, [1]]
        z = xyz_batch[:, [2]]

        depth = torch.sqrt(x ** 2 + y ** 2 + z ** 2) * max_depth
        valid = (depth > 1e-8).float()
        summary["total_data"] += len(valid)
        summary["total_valid"] += valid.sum(dim=0)  # (1,64,2048)

        r = torch.sqrt(x ** 2 + y ** 2)
        pitch = torch.atan2(z, r)
        yaw = torch.atan2(y, x)
        summary["pitch"] += torch.sum(pitch * valid, dim=0)
        summary["yaw"] += torch.sum(yaw * valid, dim=0)

    summary["pitch"] = summary["pitch"] / summary["total_valid"]
    summary["yaw"] = summary["yaw"] / summary["total_valid"]
    angles = torch.cat([summary["pitch"], summary["yaw"]], dim=0)

    mean_pitch = mean(summary["pitch"], 2).expand_as(summary["pitch"])
    mean_yaw = mean(summary["yaw"], 1).expand_as(summary["yaw"])
    mean_angles = torch.cat([mean_pitch, mean_yaw], dim=0)

    mean_valid = summary["total_valid"] / summary["total_data"]
    valid = (mean_valid > 0).float()
    angles[angles.isnan()] = 0.0
    angles = valid * angles + (1 - valid) * mean_angles

    assert angles.isnan().sum() == 0

    return angles, mean_valid


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, required=True)
    args = parser.parse_args()

    # 2D maps
    split_dirs = sorted(glob(osp.join(args.root_dir, "dataset/sequences", "*")))
    H, W = 64, 2048

    for split_dir in tqdm(split_dirs):
        point_paths = sorted(glob(osp.join(split_dir, "velodyne", "*.bin")))
        joblib.Parallel(
            n_jobs=multiprocessing.cpu_count(), verbose=10, pre_dispatch="all"
        )(
            [
                joblib.delayed(process_point_clouds)(point_path, H, W)
                for point_path in point_paths
            ]
        )

    # average angles
    # dataset = KITTIOdometry(
    #     root=osp.join(args.root_dir),
    #     split="train",
    #     shape=(H, W),
    #     is_sorted=args.sorted
    # )
    # loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=64,
    #     num_workers=4,
    #     drop_last=False,
    # )
    # N = len(dataset)

    # angles, valid = compute_avg_angles(loader)
    # torch.save(angles, osp.join(args.root_dir, "angles.pt"))

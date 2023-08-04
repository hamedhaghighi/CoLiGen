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
from util.lidar import point_cloud_to_xyz_image, labelmap
from nuscenes.nuscenes import NuScenes
import pathlib
from dataset.kitti_odometry import KITTIOdometry
from dataset.nuscene import NuScene
from collections import namedtuple
from util import make_class_from_dict

def car2hom(pc):
    return np.concatenate([pc[:, :3], np.ones((pc.shape[0], 1), dtype=pc.dtype)], axis=-1)

def image_to_pcl(rgb_image, point_cloud, velo_to_camera_rect, cam_intrinsic):
        rgb = np.zeros((len(point_cloud),3), dtype=np.int32)
        height, width, _ = rgb_image.shape
        hom_pcl_points = car2hom(point_cloud[:, :3]).T
        pcl_in_cam_rect = np.dot(velo_to_camera_rect, hom_pcl_points)
        pcl_in_image = np.dot(cam_intrinsic, pcl_in_cam_rect)
        pcl_in_image = np.array([pcl_in_image[0] / pcl_in_image[2], pcl_in_image[1] / pcl_in_image[2], pcl_in_image[2]])
        canvas_mask = (pcl_in_image[0] > 0.0) & (pcl_in_image[0] < width) & (pcl_in_image[1] > 0.0)\
            & (pcl_in_image[1] < height) & (pcl_in_image[2] > 0.0)
        valid_pcl_in_image = pcl_in_image[:, canvas_mask].astype('int32')
        rgb[canvas_mask] = rgb_image[valid_pcl_in_image[1], valid_pcl_in_image[0], :]
        return rgb

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

def load_calib(root):
        """Load and compute intrinsic and extrinsic calibration parameters."""
        # We'll build the calibration parameters as a dictionary, then
        # convert it to a namedtuple to prevent it from being modified later
        data = {}
        sequence_path = os.path.join(root, '00')
        # Load the calibration file
        calib_filepath = os.path.join(sequence_path, 'calib.txt')
        filedata = {}

        with open(calib_filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                try:
                    filedata[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        # Create 3x4 projection matrices
        P_rect_00 = np.reshape(filedata['P0'], (3, 4))
        P_rect_10 = np.reshape(filedata['P1'], (3, 4))
        P_rect_20 = np.reshape(filedata['P2'], (3, 4))
        P_rect_30 = np.reshape(filedata['P3'], (3, 4))

        data['P_rect_00'] = P_rect_00
        data['P_rect_10'] = P_rect_10
        data['P_rect_20'] = P_rect_20
        data['P_rect_30'] = P_rect_30

        # Compute the rectified extrinsics from cam0 to camN
        T1 = np.eye(4)
        T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
        T2 = np.eye(4)
        T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
        T3 = np.eye(4)
        T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

        # Compute the velodyne to rectified camera coordinate transforms
        data['T_cam0_velo'] = np.reshape(filedata['Tr'], (3, 4))
        data['T_cam0_velo'] = np.vstack([data['T_cam0_velo'], [0, 0, 0, 1]])
        data['T_cam1_velo'] = T1.dot(data['T_cam0_velo'])
        data['T_cam2_velo'] = T2.dot(data['T_cam0_velo'])
        data['T_cam3_velo'] = T3.dot(data['T_cam0_velo'])

        # Compute the camera intrinsics
        data['K_cam0'] = P_rect_00[0:3, 0:3]
        data['K_cam1'] = P_rect_10[0:3, 0:3]
        data['K_cam2'] = P_rect_20[0:3, 0:3]
        data['K_cam3'] = P_rect_30[0:3, 0:3]

        # Compute the stereo baselines in meters by projecting the origin of
        # each camera frame into the velodyne frame and computing the distances
        # between them
        p_cam = np.array([0, 0, 0, 1])
        p_velo0 = np.linalg.inv(data['T_cam0_velo']).dot(p_cam)
        p_velo1 = np.linalg.inv(data['T_cam1_velo']).dot(p_cam)
        p_velo2 = np.linalg.inv(data['T_cam2_velo']).dot(p_cam)
        p_velo3 = np.linalg.inv(data['T_cam3_velo']).dot(p_cam)

        data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
        data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)   # rgb baseline

        calib = namedtuple('CalibData', data.keys())(*data.values())
        return calib


def process_point_clouds(point_path, H, W, dest_dir, calib=None, name=None):
    is_sorted = name == 'kitti'
    def save_dir(x):
        prev_split = x.split(os.path.sep)
        seq_mode_filename = os.path.sep.join(prev_split[-4:])
        return os.path.join(dest_dir, "projected", seq_mode_filename)
    # setup point clouds
    points = np.fromfile(point_path, dtype=np.float32).reshape((-1, 4))
    # for semantic kitti
    label_path = point_path.replace("/velodyne", "/labels").replace(".bin", ".label")
    image_path = point_path.replace("/velodyne", "/image_2").replace(".bin", ".png")
    tag_path = point_path.replace("/velodyne", "/tag").replace(".bin", ".tag")
    if osp.exists(label_path):
        label = np.fromfile(label_path, dtype=np.int32)
        sem_label = label & 0xFFFF 
        if name != 'semanticPOSS':
            sem_label = _map(sem_label, labelmap)
        points = np.concatenate([points, sem_label.astype('float32')[:, None]], axis=1)
    if osp.exists(image_path):
        velo_to_camera_rect =calib.T_cam2_velo
        cam_intrinsic = calib.P_rect_20
        rgb_image = np.array(Image.open(image_path))
        rgb = image_to_pcl(rgb_image, points, velo_to_camera_rect, cam_intrinsic)
        points = np.concatenate([points, rgb.astype('float32')], axis=1)

    
    tag = np.fromfile(tag_path, dtype=np.bool) if osp.exists(tag_path) else None
    proj, _ = point_cloud_to_xyz_image(points, H, W, is_sorted=is_sorted, tag=tag)


    save_path = save_dir(point_path).replace(".bin", ".npy")
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    np.save(save_path, proj[..., :4])
    if osp.exists(label_path):
        save_path = save_dir(label_path).replace(".label", ".png")
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        labels = Image.fromarray(np.uint8(proj[..., 4]), mode="P")
        labels.putpalette(palette)
        labels.save(save_path)
    if osp.exists(image_path):
        save_path = save_dir(image_path)
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        rgb = Image.fromarray(proj[..., 5:8].astype('uint8'))
        rgb.save(save_path)


def process_nucs_point_clouds(point_path, label_path, H, W):
    filename = point_path.split(os.path.sep)[-1]
    root_dir = os.path.sep.join(point_path.split(os.path.sep)[:-1])
    root_dir = root_dir.replace('nuscene_lidarseg', 'projected_nuscene_lidarseg')
    save_dir = osp.join(root_dir, 'PCL', filename)
    label_save_dir = osp.join(root_dir, 'label', filename)
    # setup point clouds
    points = np.fromfile(point_path, dtype=np.float32).reshape((-1, 5))[:, [1, 0, 2, 3]]; points[:, 0] = -points[:, 0]
    # for semantic kitti
    if osp.exists(label_path):
        sem_label = np.fromfile(label_path, dtype=np.uint8)
        points = np.concatenate([points, sem_label.astype('float32')[:, None]], axis=1)
    proj, _ = point_cloud_to_xyz_image(points, H, W, fov_up=10.0, fov_down=-30.0, is_sorted=False)


    save_path = save_dir.replace(".bin", ".npy")
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    np.save(save_path, proj[..., :4])
    if osp.exists(label_path):
        save_path = label_save_dir.replace(".bin", ".png")
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        labels = Image.fromarray(np.uint8(proj[..., 4]))
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
        xyz_batch = item["points"]

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
    parser.add_argument("--dest-dir", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--project", action='store_true')
    args = parser.parse_args()
    DATA =  make_class_from_dict(yaml.safe_load(open(f'configs/dataset_cfg/{args.dataset_name}_cfg.yml', 'r')))
    H, W = DATA.height, DATA.width
    if args.project:
        if args.dataset_name in ['kitti', 'carla', 'semanticPOSS']:
            # calib = load_calib(osp.join(args.root_dir, "dataset/sequences"))
            calib = None
            # H, W = 64, 2048
            split_dirs = sorted(glob(osp.join(args.root_dir, "dataset/sequences", "*")))
            for split_dir in tqdm(split_dirs):
                point_paths = sorted(glob(osp.join(split_dir, "velodyne", "*.bin")))
                joblib.Parallel(
                    n_jobs=multiprocessing.cpu_count(), verbose=10, pre_dispatch="all"
                )(
                    [
                        joblib.delayed(process_point_clouds)(point_path, H, W, args.dest_dir, calib, args.dataset_name)
                        for point_path in point_paths
                    ]
                )
            
         

        elif args.dataset_name == 'nuscene':
            nusc = NuScenes(version = 'v1.0-mini', dataroot = args.root_dir, verbose = True)
            datalist = []
            labels_list=[]
            for i in range(len(nusc.sample)):
                sample = nusc.sample[i]
                sample_data_token = sample['data']['LIDAR_TOP']
                sample_path = nusc.get_sample_data_path(sample_data_token)
                label_path = (pathlib.Path(nusc.dataroot) / nusc.get("lidarseg", sample_data_token)["filename"])
                datalist.append(sample_path)
                labels_list.append(label_path)
            joblib.Parallel(
                n_jobs=multiprocessing.cpu_count(), verbose=10, pre_dispatch="all"
            )(
                [
                    joblib.delayed(process_nucs_point_clouds)(point_path, label_path, H, W)
                    for point_path, label_path in zip(datalist, labels_list)
                ]
            ) 
    else:
        if args.dataset_name in ['kitti', 'carla', 'semanticPOSS']:
            dataset = KITTIOdometry(
            args.root_dir,
            'train',
            DATA,
            shape=(H, W),
            flip=False,
            modality=['depth'],
            fill_in_label=False,
            name = args.dataset_name,
            limited_view=False)
        else:
            dataset = NuScene(
            args.root_dir,
            'train',
            None,
            shape=(32, 1024),
            flip=False,
            modality=['depth'],
            is_sorted=False,
            is_raw=True,
            fill_in_label=False)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            num_workers=4,
            drop_last=False,
        )
        # N = len(dataset)

        angles, valid = compute_avg_angles(loader)
        torch.save(angles, osp.join(args.root_dir, "angles.pt"))
    # torch.save(angles, osp.join(args.root_dir.replace('nuscene_lidarseg', 'projected_nuscene_lidarseg'), "angles.pt"))


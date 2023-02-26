"""Example of pykitti.odometry usage."""
import itertools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pykitti
import matplotlib
import glob
import os
import tqdm
import argparse

__author__ = "Lee Clement"
__email__ = "lee.clement@robotics.utias.utoronto.ca"

# Change this to the directory where you store KITTI data

color_map = {  # bgr
    0: [0, 0, 0],
    1: [0, 0, 255],
    10: [245, 150, 100],
    11: [245, 230, 100],
    13: [250, 80, 100],
    15: [150, 60, 30],
    16: [255, 0, 0],
    18: [180, 30, 80],
    20: [255, 0, 0],
    30: [30, 30, 255],
    31: [200, 40, 255],
    32: [90, 30, 150],
    40: [255, 0, 255],
    44: [255, 150, 255],
    48: [75, 0, 75],
    49: [75, 0, 175],
    50: [0, 200, 255],
    51: [50, 120, 255],
    52: [0, 150, 255],
    60: [170, 255, 150],
    70: [0, 175, 0],
    71: [0, 60, 135],
    72: [80, 240, 150],
    80: [150, 240, 255],
    81: [0, 0, 255],
    99: [255, 255, 50],
    252: [245, 150, 100],
    256: [255, 0, 0],
    253: [200, 40, 255],
    254: [30, 30, 255],
    255: [90, 30, 150],
    257: [250, 80, 100],
    258: [180, 30, 80],
    259: [255, 0, 0]
}
# Specify the dataset to load
def car2hom(pc):
    return np.concatenate([pc[:, :3], np.ones((pc.shape[0], 1), dtype=pc.dtype)], axis=-1)


def load_velo_label_scan(file):
    """Load and parse a velodyne binary file."""
    scan = np.fromfile(file, dtype=np.int32)
    return scan.reshape(-1)


def yield_velo_label_scans(velo_files):
    """Generator to parse velodyne binary files into arrays."""
    for file in velo_files:
        yield load_velo_label_scan(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("./kitti.py")
    parser.add_argument(
        '--total_samples', '-t',
        dest='total_samples',
        type=int,
        default=-1
    )
    parser.add_argument(
        '--n_seq', '-s',
        dest='n_seq',
        type=int
    )
    parser.add_argument(
        '--data_dir', '-d',
        dest='data_dir',
        type=str,
        default='/media/oem/Local Disk/Phd-datasets/dataset/'
    )
    parser.add_argument(
        '--dest_dir', '-ds',
        dest='dest_dir',
        type=str,
        default='/media/oem/Local Disk/Phd-datasets/test'
    )
    parser.add_argument(
        '--have_label', '-hl',
        dest='have_label',
        action='store_true'
    )
    
    flags, unparsed = parser.parse_known_args()
    cmap = cm.cividis
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    sample_num = flags.total_samples
    sequences_num = flags.n_seq

    basedir = flags.data_dir
    destdir = flags.dest_dir
    have_label = flags.have_label

    for seq in tqdm.tqdm(range(sequences_num), total=sequences_num):

        sequence = '{:02d}'.format(seq)
        velo_dir = os.path.join(basedir, 'sequences', sequence, 'velodyne')
        dest_velo_dir_seq = os.path.join(destdir, 'sequences', sequence, 'velodyne')
        os.makedirs(dest_velo_dir_seq, exist_ok=True)
        if have_label:
            dest_velo_label_dir_seq = os.path.join(destdir, 'sequences', sequence, 'labels')
            os.makedirs(dest_velo_label_dir_seq, exist_ok=True)

        num_frames = len(glob.glob(velo_dir + '/*'))
        n_frame_in_each_seq = min(num_frames, sample_num // sequences_num) if sample_num != -1 else num_frames
        frames = sorted(np.random.choice(np.arange(num_frames), n_frame_in_each_seq, replace=False).tolist()) if sample_num!= -1 else \
            np.arange(num_frames).tolist()
        dataset = pykitti.odometry(basedir, sequence, frames=frames)
        if have_label:
            velo_label_files = sorted(glob.glob(
                os.path.join(basedir, 'sequences', sequence,'labels', '*.label')))
            velo_label_files = [velo_label_files[i] for i in frames]
            velo_label = yield_velo_label_scans(velo_label_files)

        for pc, img, fr in tqdm.tqdm(zip(dataset.velo, dataset.cam2, frames), total=len(frames)):

            label = next(velo_label) if have_label else None
            if have_label:
                assert label.shape[0] == pc.shape[0]
            img = np.array(img)
            height, width, _ = img.shape
            velo_to_camera_rect = dataset.calib.T_cam2_velo
            cam_intrinsic = dataset.calib.P_rect_20
            pc_points = car2hom(pc).T
            pc_in_cam_rect = np.dot(velo_to_camera_rect, pc_points)
            pc_in_image = np.dot(cam_intrinsic, pc_in_cam_rect)
            pc_in_image = np.array([pc_in_image[0] / pc_in_image[2], pc_in_image[1] / pc_in_image[2], pc_in_image[2]])
            canvas_mask = (pc_in_image[0] > 0.0) & (pc_in_image[0] < width) & (pc_in_image[1] > 0.0)\
                & (pc_in_image[1] < height) & (pc_in_image[2] > 0.0)
            
            coords = pc_in_image[:, canvas_mask].astype('int32')
            pc = pc[canvas_mask]
            pc_rgb = img[coords[1], coords[0]].astype('float32')
            pc_with_rgb = np.concatenate([pc, pc_rgb], axis=-1)


            dest_velo_filename = os.path.join(dest_velo_dir_seq, '{:06d}.bin'.format(fr))
            pc_with_rgb.reshape(-1).tofile(dest_velo_filename)

            if label is not None:
                label = label[canvas_mask]
                dest_velo_label_filename = os.path.join(dest_velo_label_dir_seq, '{:06d}.label'.format(fr))
                label.tofile(dest_velo_label_filename)

            # dot_extent = 1
            # if label is not None:
            #     colors = np.array([color_map[v & 0xFFFF] for v in label]).astype('uint8')
            # else:
            #     colors = np.array([[255, 255, 0] for _ in range(pc_with_rgb.shape[0])]).astype('uint8')

            # for i in range(coords.shape[1]):
            #         img[
            #             coords[1][i]-dot_extent: coords[1][i]+dot_extent,
            #             coords[0][i]-dot_extent: coords[0][i]+dot_extent] = colors[i]
            # plt.imshow(img)
            # plt.show()



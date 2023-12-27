import numpy as np
import matplotlib.pyplot as plt
import yaml
import os.path as osp
from glob import glob
import numpy as np
from tqdm import tqdm
from util import _map
from PIL import Image

def load_datalist(root):
    subsets = range(10)
    datalist_label = []
    datalist_points = []
    for subset in subsets:
        subset_dir = osp.join(root, str(subset).zfill(2))
        sub_label_path = sorted(glob(osp.join(subset_dir, "labels/*")))
        sub_point_path = sorted(glob(osp.join(subset_dir, "velodyne/*")))
        datalist_label += list(sub_label_path)
        datalist_points += list(sub_point_path)
    return np.array(datalist_label), np.array(datalist_points)



def main():
    # Replace these paths with the actual paths to the KITTI dataset on your machine
    is_raw = False
    np.random.seed(0)
    dataset_name = 'carla'
    ds_cfg = yaml.safe_load(open(f'configs/dataset_cfg/{dataset_name}_cfg.yml', 'r'))
    data_dir = osp.join(ds_cfg['data_dir'], "sequences")
    label_id_list = list(ds_cfg['learning_map'].keys())
    cm = plt.get_cmap('gist_rainbow')
    hist = dict()
    label_path_list, point_path_list = load_datalist(data_dir)
    idx_array = np.arange(len(label_path_list))
    np.random.shuffle(idx_array)
    label_path_list, point_path_list = label_path_list[idx_array][:5000], point_path_list[idx_array][:5000]
    mu, std = np.zeros(5), np.zeros(5)
    n_points = 0
    max_depth = 0
    for p_l, l_p in tqdm(zip(point_path_list,label_path_list)):
        if is_raw:
            point_cloud = np.fromfile(p_l, dtype=np.float32).reshape((-1, 4))
            depth = np.linalg.norm(point_cloud[:, :3], ord=2, axis=1)
        else:
            point_cloud = np.load(p_l).astype(np.float32)
            depth = point_cloud[0].reshape(-1)
            point_cloud = np.transpose(point_cloud[1:5].reshape(4, -1), (1,0))
        n_points += len(point_cloud)
        max_depth = max(max_depth, np.max(depth))
        mu[0] += depth.sum()
        mu[1:] += point_cloud.sum(axis=0)
        std[0] += (depth** 2).sum() 
        std[1:] += (point_cloud** 2).sum(axis=0)
        if is_raw:
            label_id_array = np.fromfile(l_p, dtype=np.int32)
            label_id_array = label_id_array & 0xFFFF
            label_id_array = _map(label_id_array, ds_cfg['learning_map'])
            label_id_array = _map(label_id_array, ds_cfg['learning_map_inv'])
        else:
            label_id_array = np.array(Image.open(l_p))
            label_id_array = _map(label_id_array, ds_cfg['learning_map_inv'])
        for id in label_id_list:
            s = (label_id_array == id).sum()
            if s > 0:
                if id in hist:
                    hist[id] += s
                else:
                    hist[id] = s
    mu = mu/n_points
    std = np.sqrt((std/n_points) - mu**2)
    print('max_depth:', max_depth)
    print('mu:', mu)
    print('std:', std)
    print('hist', {k:v/n_points for k ,v in hist.items()})
    # # Plot the histogram
    # num_label =len(hist)
    # color_list = [cm(i/num_label) for i in range(num_label)]
    # np.random.shuffle(color_list)
    # classes = np.arange(num_label)
    # for i, (k , v) in enumerate(hist.items()):
    #     bar = plt.bar([i], [v], label=k)
    #     bar[0].set_color(color_list[i])
    # plt.xlabel('Semantic Label')
    # plt.ylabel('Frequency')
    # plt.title(f'Histogram of Semantic Labels in {dataset_name} Dataset')
    # plt.xticks(classes)
    # plt.legend()
    # ax = plt.gca()
    # leg = ax.get_legend()
    # for i, lgh in enumerate(leg.legendHandles):
    #     lgh.set_color(color_list[i])
    # plt.show()

if __name__ == "__main__":
    main()

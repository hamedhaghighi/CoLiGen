import numpy as np
import matplotlib.pyplot as plt
import yaml
import os.path as osp
from glob import glob
import numpy as np
from tqdm import tqdm
from util import _map

def load_datalist(root):
    subsets = range(10)
    datalist = []
    for subset in subsets:
        subset_dir = osp.join(root, str(subset).zfill(2))
        sub_point_paths = sorted(glob(osp.join(subset_dir, "labels/*")))
        datalist += list(sub_point_paths)
    return datalist



def main():
    # Replace these paths with the actual paths to the KITTI dataset on your machine
    np.random.seed(0)
    dataset_name = 'carla'
    ds_cfg = yaml.safe_load(open(f'configs/dataset_cfg/{dataset_name}_cfg.yml', 'r'))
    data_dir = osp.join(ds_cfg['data_dir'], "sequences")
    label_id_list = list(ds_cfg['labels'].keys())
    cm = plt.get_cmap('gist_rainbow')


    hist = dict()
    label_path_list = load_datalist(data_dir)
    for l_p in tqdm(label_path_list):
        label_id_array = np.fromfile(l_p, dtype=np.int32)
        label_id_array = label_id_array & 0xFFFF
        label_id_array = _map(label_id_array, ds_cfg['learning_map'])
        label_id_array = _map(label_id_array, ds_cfg['learning_map_inv'])
        for id in label_id_list:
            s = (label_id_array == id).sum()
            if s > 0:
                if ds_cfg['labels'][id] in hist:
                    hist[ds_cfg['labels'][id]] += s
                else:
                    hist[ds_cfg['labels'][id]] = s
        
    # Plot the histogram
    num_label = len(list(hist.keys()))
    color_list = [cm(i/num_label) for i in range(num_label)]
    np.random.shuffle(color_list)
    classes = np.arange(num_label)
    for i, (k , v) in enumerate(hist.items()):
        bar = plt.bar([i], [v], label=k)
        bar[0].set_color(color_list[i])
    plt.xlabel('Semantic Label')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Semantic Labels in {dataset_name} Dataset')
    plt.xticks(classes)
    plt.legend()
    ax = plt.gca()
    leg = ax.get_legend()
    for i, lgh in enumerate(leg.legendHandles):
        lgh.set_color(color_list[i])
    plt.show()

if __name__ == "__main__":
    main()

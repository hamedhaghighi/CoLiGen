"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time

from matplotlib import cm
from numpy.lib.shape_base import split
# from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.fid import FID
from util.util import map, tensor2im
from dataset.datahandler import Loader, deg2rad
import yaml
import argparse
import numpy as np
import torch
from tqdm import trange
import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import os

def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def deg2rad(theta):
  return theta / 180.0 * np.pi

class M_parser():
    def __init__(self, cfg_path, data_dir):
        opt_dict = yaml.safe_load(open(cfg_path, 'r'))
        for k , v in opt_dict.items():
            setattr(self, k, v)
        if data_dir != '':
            self.dataset['dataset_A']['data_dir'] = data_dir
        self.isTrain = False


def save_data(data_dir, exp_name, test_image_results):

    splitted_data_dir = data_dir.split('/')
    root_dir = '/'.join(splitted_data_dir[:-1])
    dataset_name = splitted_data_dir[-1]
    new_data_dir = os.path.join(root_dir, 'synthesized_' + exp_name + '_' + dataset_name, 'sequences', '00')
    scan_dir = os.path.join(new_data_dir, 'velodyne')
    label_dir = os.path.join(new_data_dir, 'labels')
    os.makedirs(scan_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    real_A = test_image_results['real_A']
    points = test_image_results['points']
    proj_idx = test_image_results['proj_idx']
    sem_label = test_image_results['sem_label']
    y_ind = np.arange(0, 256, 4)
    proj_remission = test_image_results['fake_B']
    proj_remission = [p[:,y_ind] for p in proj_remission]
    for i in range(len(points)):
        mask = proj_idx[i] > 0
        selected_points = points[i][proj_idx[i][mask]]
        selected_remission = proj_remission[i][mask].reshape(-1,1)
        p_cloud = np.concatenate((selected_points, selected_remission), axis=-1)
        scan_filename = os.path.join(scan_dir, '{0: 06}.bin'.format(i))
        p_cloud.reshape(-1).astype('float32').tofile(scan_filename)
        if sem_label is not None:
            selected_labels = sem_label[i][proj_idx[i][mask]]
            scan_label_filename = os.path.join(label_dir, '{0: 06}.label'.format(i))
            selected_labels.reshape(-1).astype('int32').tofile(scan_label_filename)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_test', type=str, help='Path of the config file')
    parser.add_argument('--data_dir', type=str, default='', help='Path of the dataset')
    parser.add_argument('--is_train_data', '-it' , action='store_true',  help='is train data')

    pa = parser.parse_args()
    opt = M_parser(pa.cfg_test, pa.data_dir)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    # DATA = yaml.safe_load(open(pa.cfg_dataset, 'r'))
    ## test whole code fast

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    g_steps = 0
    KL = Loader(data_dict=opt.dataset, batch_size=opt.batch_size,\
         val_split_ratio=opt.val_split_ratio, max_dataset_size=opt.max_dataset_size,test_dataset_size=opt.test_dataset_size, workers= opt.n_workers, is_train=False,
          is_training_data=pa.is_train_data)

    fid_cls = FID(KL.total_dataset, opt.fid_stats_path)

    e_steps = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
    visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
    test_dl = iter(KL.testloader)
    n_test_batch = len(KL.testloader)

    test_losses = defaultdict(list)
    test_image_results = defaultdict(list)
    model.train(False)
    tq = tqdm.tqdm(total=n_test_batch, desc='val_Iter', position=5)
    n_pics = 0
    generated_remission = []
    for i in range(n_test_batch):
        data = next(test_dl)
        model.set_input_PCL(data)
        with torch.no_grad():
            model.evaluate_model()
        for k ,v in model.get_current_losses(is_eval=True).items():
            test_losses[k].append(v)

        vis_dict = model.get_current_visuals()
        if opt.fid_dataset:
            generated_remission.append(data[2].cpu().detach())
        else:
            generated_remission.append(vis_dict['fake_B'].cpu().detach())
        for k, v in vis_dict.items():
            test_image_results[k].append(v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v)
        tq.update(1)

    # test_image_results = {k: listnp.concatenate(v, axis=0) for k, v in test_image_results.items()}
    test_image_results_t = defaultdict(list)
    for k , v in test_image_results.items():
        val_list = []
        for vp in v:
            val_list.extend(vp)
        test_image_results_t[k] = val_list
    fid_score = fid_cls.fid_score(generated_remission)
    losses = {k: np.array(v).mean() for k , v in test_losses.items()}
    print (losses)
    print('FID score: ', fid_score)

    if opt.save_generated_data:
        
        data_dir = opt.dataset['dataset_A']['data_dir']
        save_data(data_dir, opt.name, test_image_results_t)

    ### visualise images

    def subsample(img):
        # img shape C, H , W
        if len(img.shape) == 3:
            _, H , _ = img.shape
        elif len(img.shape) == 2:
            H, _ = img.shape
        y_ind = np.arange(0, H, 4)
        if len(img.shape) == 3:
            return img[:, y_ind, :] * 0.5 + 0.5
        return img[y_ind, :] * 0.5 + 0.5

    def save_img(img, tag, pic_dir, cmap=None):
        fig = plt.figure()
        if cmap is not None:
            plt.imshow(img, cmap=cmap, vmin=0.0, vmax=1.0)
        else:
            plt.imshow(img)
        plt.axis('off')
        fname = os.path.join(pic_dir, 'img_' + tag + '.png')
        plt.savefig(fname, pad_inches=0)
        plt.close(fig)
        
    visual_keys = ['fake_B', 'real_B', 'range', 'real_A']
    exp_name = os.path.join(opt.checkpoints_dir, opt.name, 'test_results_pics')
    os.makedirs(exp_name, exist_ok=True)
    n_pics = opt.test_dataset_size
    for i in range(n_pics):
        pic_dir = os.path.join(exp_name, 'img_' + str(i))
        os.makedirs(pic_dir , exist_ok=True)
        
        for k, img in test_image_results_t.items():
            if k in visual_keys:
                if k == 'real_A':
                    if img[i].shape[1] == 6:
                        rgb = img[i][3:]
                        save_img(subsample(rgb).transpose((1, 2, 0)), 'rgb', pic_dir)
                    img[i] = img[i][:3]

                cmap = 'inferno' if k == 'range' else 'cividis'
                for j in range(img[i].shape[0]):
                    save_img(subsample(img[i][j]), k+str(j), pic_dir, cmap)
        
        

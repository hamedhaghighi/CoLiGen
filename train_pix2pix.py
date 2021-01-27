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
# from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from dataset.ProjectedKitti import Kitti_Loader
import yaml
import argparse
import numpy as np
import torch
from tqdm import trange
import tqdm
from collections import defaultdict

class M_parser():
    def __init__(self, cfg_path):
        opt_dict = yaml.safe_load(open(cfg_path, 'r'))
        for k , v in opt_dict.items():
            setattr(self, k, v)
        self.isTrain = True
        self.epoch_decay = self.n_epochs//2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_train', type=str, help='Path of the config file')
    parser.add_argument('--cfg_dataset', type=str, help='Path of the config file')
    pa = parser.parse_args()
    opt = M_parser(pa.cfg_train)
    DATA = yaml.safe_load(open(pa.cfg_dataset, 'r'))
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    g_steps = 0
    KL = Kitti_Loader(data_dir=opt.data_dir, batch_size=opt.batch_size, data_stats=opt.dataset['sensor'], val_slpit_ratio=opt.val_split_ratio)
    train_dl = iter(KL.trainloader)
    valid_dl = iter(KL.validloader)
    if opt.fast_test:
        n_train_batch = 2
        n_valid_batch = 2
        opt.n_epochs = 2
        opt.epoch_decay = opt.n_epochs//2
    else:
        n_train_batch = len(train_dl)
        n_valid_batch = len(valid_dl)
    epoch_tq = tqdm.tqdm(total=opt.n_epochs, desc='Epoch', position=1)
    start_from_epoch = model.schedulers[0].last_epoch if opt.continue_train else 0 
    for epoch in range(start_from_epoch, opt.n_epochs):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        e_steps = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        model.train(True)
        train_tq = tqdm.tqdm(total=n_train_batch, desc='Iter', position=3)
        for i in range(n_train_batch):  # inner loop within one epoch
            data = next(train_dl)
            iter_start_time = time.time()  # timer for computation per iteration
            g_steps += 1
            e_steps += 1
            model.set_input_PCL(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if g_steps % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                visualizer.display_current_results('train', model.get_current_visuals(), g_steps)

            if g_steps % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses('train', epoch, e_steps, losses, train_tq)
                visualizer.plot_current_losses('train', epoch, losses, g_steps)

            if g_steps % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                train_tq.write('saving the latest model (epoch %d, total_iters %d)' % (epoch, g_steps))
                save_suffix = 'latest'
                model.save_networks(save_suffix)
            train_tq.update(1)
        val_losses = defaultdict(list)
        model.train(False)
        val_tq = tqdm.tqdm(total=n_valid_batch, desc='val_Iter', position=5)
        for i in range(n_valid_batch):
            data = next(valid_dl)
            model.set_input_PCL(data)
            with torch.no_grad():
                model.evaluate_model()
            for k ,v in model.get_current_losses().items():
                val_losses[k].append(v)
            val_tq.update(1)
        losses = {k: np.array(v).mean() for k , v in val_losses.items()}
        visualizer.plot_current_losses('val', epoch, losses, g_steps)
        visualizer.display_current_results('val', model.get_current_visuals(), g_steps)
        visualizer.print_current_losses('val', epoch, e_steps, losses, val_tq)

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, g_steps))
            model.save_networks('latest')
            model.save_networks(epoch)
        epoch_tq.update(1)

        print('End of epoch %d \t Time Taken: %d sec' % (epoch, time.time() - epoch_start_time))

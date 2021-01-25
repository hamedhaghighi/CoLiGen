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
from dataset.Kitti import Kitti_Loader
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
    g_steps = 0                # the total number of training iterations
    KL = Kitti_Loader(root=opt.data_dir,
                                    train_sequences=DATA["split"]["train"],
                                    valid_sequences=DATA["split"]["valid"],
                                    test_sequences=None,
                                    labels=DATA["labels"],
                                    color_map=DATA["color_map"],
                                    learning_map=DATA["learning_map"],
                                    learning_map_inv=DATA["learning_map_inv"],
                                    sensor=opt.dataset["sensor"],
                                    max_points=opt.dataset["max_points"],
                                    batch_size=opt.batch_size,
                                    workers= 4,
                                    gt=True,
                                    shuffle_train=True)
    train_dl = iter(KL.trainloader)
    valid_dl = iter(KL.validloader)
    dataset_size = len(train_dl) * opt.batch_size
    if opt.fast_test:
        n_train_batch = 10
        n_valid_batch = 9
        opt.n_epochs = 10
        opt.n_epochs_decay = 0
    else:
        n_train_batch = len(train_dl)
        n_valid_batch = len(valid_dl)
    epoch_tq = tqdm.tqdm(total=opt.n_epochs + opt.n_epochs_decay, desc='Epoch', position=1)
    for epoch in range(opt.n_epochs + opt.n_epochs_decay):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        e_steps = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        model.train(True)
        train_tq = tqdm.tqdm(total=n_train_batch, desc='Iter', position=3)
        for i in range(n_train_batch):  # inner loop within one epoch
            data = next(train_dl)
            proj_xyz , proj_remission, proj_range = data
            iter_start_time = time.time()  # timer for computation per iteration
            if g_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            g_steps += 1
            e_steps += 1
            model.set_input_PCL(proj_xyz, proj_remission, proj_range)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if g_steps % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                visualizer.display_current_results('train', model.get_current_visuals(), g_steps)

            if g_steps % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses('train', epoch, e_steps, losses, train_tq)
                visualizer.plot_current_losses('train', epoch, losses, g_steps)

            if g_steps % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                train_tq.write('saving the latest model (epoch %d, total_iters %d)' % (epoch, g_steps))
                save_suffix = 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
            train_tq.update(1)
        val_losses = defaultdict(list)
        model.train(False)
        val_tq = tqdm.tqdm(total=n_valid_batch, desc='val_Iter', position=5)
        for i in range(n_valid_batch):
            data = next(valid_dl)
            proj_xyz, proj_remission, proj_range = data
            model.set_input_PCL(proj_xyz, proj_remission, proj_range)
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

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs_decay, time.time() - epoch_start_time))

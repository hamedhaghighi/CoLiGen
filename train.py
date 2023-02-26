import time
# from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.fid import FID
from dataset.datahandler import get_data_loader
import yaml
import argparse
import numpy as np
import torch
from tqdm import trange
import tqdm
import os
from util.lidar import LiDAR
from collections import defaultdict


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def make_class_from_dict(opt):
    if any([isinstance(k, int) for k in opt.keys()]):
        return opt
    else:
        class dict_class():
            def __init__(self):
                for k , v in opt.items():
                    if isinstance(v , dict):
                        setattr(self, k, make_class_from_dict(v)) 
                    else:
                        setattr(self, k, v)
        return dict_class()

class M_parser():
    def __init__(self, cfg_path, data_dir):
        opt_dict = yaml.safe_load(open(cfg_path, 'r'))
        dict_class = make_class_from_dict(opt_dict)
        members = [attr for attr in dir(dict_class) if not callable(getattr(dict_class, attr)) and not attr.startswith("__")]
        for m in members:
            setattr(self, m, getattr(dict_class, m))
        if data_dir != '':
            self.dataset.dataset_A.data_dir = data_dir
        self.model.isTrain = True
        self.training.isTrain = True
        self.training.epoch_decay = self.training.n_epochs//2



def modify_opt_for_fast_test(opt):
    opt.n_epochs = 2 
    opt.epoch_decay = opt.n_epochs//2
    opt.display_freq = 1
    opt.print_freq = 1
    opt.save_latest_freq = 100
    opt.max_dataset_size = 10
    opt.batch_size = 2
    opt.name = 'test'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_train', type=str, help='Path of the config file')
    parser.add_argument('--data_dir', type=str, default='', help='Path of the dataset')
    

    cl_args = parser.parse_args()
    opt = M_parser(cl_args.cfg_train, cl_args.data_dir)
    torch.manual_seed(opt.training.seed)
    np.random.seed(opt.training.seed)
    # DATA = yaml.safe_load(open(pa.cfg_dataset, 'r'))
    ## test whole code fast
    if opt.training.fast_test:
        modify_opt_for_fast_test(opt.training)

    lidar = LiDAR(
    num_ring=opt.dataset.dataset_A.img_prop.width,
    num_points=opt.dataset.dataset_A.img_prop.height,
    min_depth=opt.dataset.dataset_A.min_depth,
    max_depth=opt.dataset.dataset_A.max_depth,
    angle_file=os.path.join(opt.dataset.dataset_A.data_dir, "angles.pt"),
    )
    # lidar.to(device)
    model = create_model(opt, lidar)      # create a model given opt.model and other options
    model.setup(opt.training)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt.training, lidar)   # create a visualizer that display/save images and plots
    g_steps = 0

    train_dl, train_dataset = get_data_loader(opt.dataset, 'train', opt.training.batch_size)
    val_dl, _ = get_data_loader(opt.dataset, 'val', opt.training.batch_size)

    fid_cls = FID(train_dataset, opt.dataset.dataset_A.data_dir) if opt.training.calc_FID else None

    epoch_tq = tqdm.tqdm(total=opt.training.n_epochs, desc='Epoch', position=1)
    start_from_epoch = model.schedulers[0].last_epoch if opt.training.continue_train else 0 
    data_maps = opt.dataset.dataset_A
    #### Train & Validation Loop
    for epoch in range(start_from_epoch, opt.training.n_epochs):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        e_steps = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        model.train(True)
        train_dl = iter(train_dl)
        valid_dl = iter(val_dl)
        n_train_batch = len(train_dl)
        n_valid_batch = len(val_dl)
        
        train_tq = tqdm.tqdm(total=n_train_batch, desc='Iter', position=3)
        for _ in range(n_train_batch):  # inner loop within one epoch
            data = next(train_dl)
            iter_start_time = time.time()  # timer for computation per iteration
            g_steps += 1
            e_steps += 1
            model.set_input_PCL(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if g_steps % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                visualizer.display_current_results('train', model.get_current_visuals(), g_steps, data_maps)

            if g_steps % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses(is_eval=True)
                visualizer.print_current_losses('train', epoch, e_steps, losses, train_tq)
                visualizer.plot_current_losses('train', epoch, losses, g_steps)

            if g_steps % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                train_tq.write('saving the latest model (epoch %d, total_iters %d)' % (epoch, g_steps))
                save_suffix = 'latest'
                model.save_networks(save_suffix)
            train_tq.update(1)

        ##### Do validation ....    
        val_losses = defaultdict(list)
        model.train(False)
        val_tq = tqdm.tqdm(total=n_valid_batch, desc='val_Iter', position=5)
        dis_batch_ind = np.random.randint(0, n_valid_batch)
        generated_remission = []
        for i in range(n_valid_batch):
            data = next(valid_dl)
            model.set_input_PCL(data)
            with torch.no_grad():
                model.evaluate_model()

            vis_dict = model.get_current_visuals()
            if fid_cls is not None:
                generated_remission.append(vis_dict['fake_B'].detach().cpu())

            if i == dis_batch_ind:
                visualizer.display_current_results('val', vis_dict, g_steps, data_maps)

            for k ,v in model.get_current_losses(is_eval=True).items():
                val_losses[k].append(v)
            val_tq.update(1)
        
        if fid_cls is not None:
            fid_score = fid_cls.fid_score(generated_remission, batch_size= opt.batch_size)
            visualizer.plot_current_losses('val', epoch, {'FID':fid_score}, g_steps)

        losses = {k: np.array(v).mean() for k , v in val_losses.items()}
        
        visualizer.plot_current_losses('val', epoch, losses, g_steps)
        visualizer.print_current_losses('val', epoch, e_steps, losses, val_tq)
        epoch_tq.update(1)

        print('End of epoch %d \t Time Taken: %d sec' % (epoch, time.time() - epoch_start_time))

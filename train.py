import time
# from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from fid import FID
from dataset.datahandler import get_data_loader
import yaml
import argparse
import numpy as np
import torch
import tqdm
import os
from util.lidar import LiDAR
from util import *
from collections import defaultdict
import shutil
from util.sampling.fps import downsample_point_clouds
from util.metrics.cov_mmd_1nna import compute_cov_mmd_1nna
from util.metrics.jsd import compute_jsd
from util.metrics.swd import compute_swd
os.environ['LD_PRELOAD'] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def inv_to_xyz(inv, lidar, tol=1e-8):
        inv = tanh_to_sigmoid(inv).clamp_(0, 1)
        xyz = lidar.inv_to_xyz(inv, tol)
        xyz = xyz.flatten(2).transpose(1, 2)  # (B,N,3)
        xyz = downsample_point_clouds(xyz, 512)
        return xyz

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
        self.model.isTrain = self.training.isTrain = not self.training.test
        self.training.epoch_decay = self.training.n_epochs//2



def modify_opt_for_fast_test(opt):
    opt.n_epochs = 1
    opt.epoch_decay = opt.n_epochs//2
    opt.display_freq = 1
    opt.print_freq = 1
    opt.save_latest_freq = 1
    opt.max_dataset_size = 10
    opt.batch_size = 1


def check_exp_exists(opt, cfg_args):
    cfg_path = cfg_args.cfg_train
    opt_t = opt.training
    opt_m = opt.model
    opt_d = opt.dataset.dataset_A
    modality_A = '_'.join(opt_m.modality_A)
    out_ch = ''

    for k in [attr for attr in dir(opt_m.out_ch) if not attr.startswith("__")]:
        out_ch += f'{k}_{getattr(opt_m.out_ch, k)}_'
    if cfg_args.load:
        opt_t.name = cfg_path.split(os.sep)[1]
    elif cfg_args.fast_test:
        opt_t.name = 'test'
    else:
        opt_t.name = f'modality_A_{modality_A}_out_ch_{out_ch}_L_L1_{opt_m.lambda_L1}' \
            + f'_L_GAN_{opt_m.lambda_LGAN}_L_mask_{opt_m.lambda_mask}_w_{opt_d.img_prop.width}_h_{opt_d.img_prop.height}' \
                + f'_netG_{opt_m.netG}_netD_{opt_m.netD}_batch_size_{opt_t.batch_size}'
        
    exp_dir = os.path.join(opt_t.checkpoints_dir, opt_t.name)
    if not opt_t.continue_train and opt_t.isTrain:
        if os.path.exists(exp_dir):
            reply = ''
            
            while not reply.startswith('y') and not reply.startswith('n'):
                reply = str(input(f'exp_dir {exp_dir} exists. Do you want to delete it? (y/n): \n')).lower().strip()
            if reply.startswith('y'):
                shutil.rmtree(exp_dir)
            else:
                print('Please Re-run the program with \"continue train\" enabled')
                exit(0)
        os.makedirs(exp_dir, exist_ok=True)
        shutil.copy(cfg_path, exp_dir)
    else:
        assert os.path.exists(exp_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_train', type=str, help='Path of the config file')
    parser.add_argument('--data_dir', type=str, default='', help='Path of the dataset')
    parser.add_argument('--load', action='store_true', help='if load true the exp name comes from checkpoint path')
    parser.add_argument('--fast_test', action='store_true', help='fast test of experiment')
    parser.add_argument('--fid_dataset_name', type=str, default='', help='fast test of experiment')


    cl_args = parser.parse_args()
    opt = M_parser(cl_args.cfg_train, cl_args.data_dir)
    torch.manual_seed(opt.training.seed)
    np.random.seed(opt.training.seed)
    # DATA = yaml.safe_load(open(pa.cfg_dataset, 'r'))
    ## test whole code fast
    if cl_args.fast_test and opt.training.isTrain:
        modify_opt_for_fast_test(opt.training)

    if not opt.training.isTrain:
        opt.training.n_epochs = 1

    check_exp_exists(opt, cl_args)

    device = torch.device('cuda:{}'.format(opt.training.gpu_ids[0])) if opt.training.gpu_ids else torch.device('cpu') 
    lidar = LiDAR(
    num_ring=opt.dataset.dataset_A.img_prop.height,
    num_points=opt.dataset.dataset_A.img_prop.width,
    angle_file=os.path.join(opt.dataset.dataset_A.data_dir, "angles.pt"),
    dataset_name=opt.dataset.dataset_A.name
    )
    lidar.to(device)
    model = create_model(opt, lidar)      # create a model given opt.model and other options
    model.setup(opt.training)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt.training, lidar, dataset_name=opt.dataset.dataset_A.name)   # create a visualizer that display/save images and plots
    g_steps = 0

    train_dl, train_dataset = get_data_loader(opt.dataset, 'train', opt.training.batch_size, dataset_name=opt.dataset.dataset_A.name)
    val_dl, _ = get_data_loader(opt.dataset, 'val' if opt.training.isTrain else 'test', opt.training.batch_size, dataset_name=opt.dataset.dataset_A.name)  
    fid_cls = FID(train_dataset, cl_args.fid_dataset_name, lidar) if cl_args.fid_dataset_name!= '' else None

    epoch_tq = tqdm.tqdm(total=opt.training.n_epochs, desc='Epoch', position=1)
    start_from_epoch = model.schedulers[0].last_epoch if opt.training.continue_train else 0 
    data_maps = opt.dataset.dataset_A
    #### Train & Validation Loop
    for epoch in range(start_from_epoch, opt.training.n_epochs):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        e_steps = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        # Train loop
        if opt.training.isTrain:
            visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
            model.update_learning_rate()    # update learning rates in the beginning of every epoch.
            model.train(True)
            train_dl_iter = iter(train_dl)
            n_train_batch = 2 if cl_args.fast_test else len(train_dl)
            train_tq = tqdm.tqdm(total=n_train_batch, desc='Iter', position=3)
            for _ in range(n_train_batch):  # inner loop within one epoch
                data = next(train_dl_iter)
                # import matplotlib.pyplot as plt
                # plt.figure(0)
                # plt.imshow(np.clip(data['depth'][0,0].numpy()* 5, 0, 1))
                # plt.figure(1)
                # plt.imshow(np.clip(data['reflectance'][0,0].numpy(),0 ,1))
                # plt.figure(2)
                # plt.imshow(data['label'][0,0].numpy())
                # plt.show()
                iter_start_time = time.time()  # timer for computation per iteration
                g_steps += 1
                e_steps += 1
                model.set_input_PCL(data)         # unpack data from dataset and apply preprocessing
                model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

                if g_steps % opt.training.display_freq == 0:   # display images on visdom and save images to a HTML file
                    visualizer.display_current_results('train', model.get_current_visuals(), g_steps, data_maps)

                if g_steps % opt.training.print_freq == 0:    # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    visualizer.print_current_losses('train', epoch, e_steps, losses, train_tq)
                    visualizer.plot_current_losses('train', epoch, losses, g_steps)

                if g_steps % opt.training.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                    train_tq.write('saving the latest model (epoch %d, total_iters %d)' % (epoch, g_steps))
                    save_suffix = 'latest'
                    model.save_networks(save_suffix)
                train_tq.update(1)

        val_dl_iter = iter(val_dl)
        n_val_batch = 2 if cl_args.fast_test else  len(val_dl)
        ##### validation
        val_losses = defaultdict(list)
        model.train(False)
        tag = 'val' if opt.training.isTrain else 'test'
        val_tq = tqdm.tqdm(total=n_val_batch, desc='val_Iter', position=5)
        dis_batch_ind = np.random.randint(0, n_val_batch)
        data_dict = defaultdict(list)
        fid_samples = [] if fid_cls is not None else None
        for i in range(n_val_batch):
            data = next(val_dl_iter)
            model.set_input_PCL(data)
            with torch.no_grad():
                model.evaluate_model()

            vis_dict = model.get_current_visuals()

            data_dict['synth-2d'].append(model.synth_inv)
            data_dict['synth-3d'].append(inv_to_xyz(model.synth_inv, lidar))

            if fid_cls is not None:
                synth_depth = lidar.revert_depth(tanh_to_sigmoid(model.synth_inv), norm=False)
                synth_points = lidar.inv_to_xyz(tanh_to_sigmoid(model.synth_inv)) * lidar.max_depth
                synth_reflectance = tanh_to_sigmoid(model.synth_reflectance)
                synth_data = torch.cat([synth_depth, synth_points, synth_reflectance, model.synth_mask], dim=1)
                fid_samples.append(synth_data)



            if i == dis_batch_ind:
                visualizer.display_current_results(tag, vis_dict, g_steps, data_maps)

            for k ,v in model.get_current_losses(is_eval=True).items():
                val_losses[k].append(v)
            val_tq.update(1)
        
        # if fid_cls is not None:
        #     fid_score = fid_cls.fid_score(generated_remission, batch_size= opt.batch_size)
        #     visualizer.plot_current_losses('val', epoch, {'FID':fid_score}, g_steps)

        losses = {k: float(np.array(v).mean()) for k , v in val_losses.items()}
        visualizer.plot_current_losses(tag, epoch, losses, g_steps)
        visualizer.print_current_losses(tag, epoch, e_steps, losses, val_tq)
        test_dl, test_dataset = get_data_loader(opt.dataset, 'test', opt.training.batch_size, dataset_name=opt.dataset.dataset_A.name)
        test_dl_iter = iter(test_dl)
        n_test_batch = 2 if cl_args.fast_test else  len(test_dl)
        N = n_test_batch * opt.training.batch_size if cl_args.fast_test else len(test_dataset)
        ##### calculating unsupervised metrics
        test_tq = tqdm.tqdm(total=n_test_batch, desc='real_data', position=5)
        for i in range(len(test_dl)):
            data = next(test_dl_iter)
            data = fetch_reals(data, lidar, device)
            data_dict['real-2d'].append(data['inv'])
            data_dict['real-3d'].append(inv_to_xyz(data['inv'], lidar))
            test_tq.update(1)

        for k ,v in data_dict.items():
            data_dict[k] = torch.cat(v, dim=0)[: N]
        scores = {}
        scores.update(compute_swd(data_dict["synth-2d"], data_dict["real-2d"]))
        scores["jsd"] = compute_jsd(data_dict["synth-3d"] / 2.0, data_dict["real-3d"] / 2.0)
        scores.update(compute_cov_mmd_1nna(data_dict["synth-3d"], data_dict["real-3d"], 512, ("cd",)))
        if fid_cls is not None:
            scores['fid'] = fid_cls.fid_score(torch.cat(fid_samples, dim=0))
        visualizer.plot_current_losses('unsupervised_metrics', epoch, scores, g_steps)
        visualizer.print_current_losses('unsupervised_metrics', epoch, e_steps, scores, val_tq)

        epoch_tq.update(1)

        print('End of epoch %d \t Time Taken: %d sec' % (epoch, time.time() - epoch_start_time))

import time

from matplotlib import cm
# from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.fid import FID
from dataset.datahandler import Loader
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


class M_parser():
    def __init__(self, cfg_path, data_dir):
        opt_dict = yaml.safe_load(open(cfg_path, 'r'))
        for k , v in opt_dict.items():
            setattr(self, k, v)
        if data_dir != '':
            self.dataset['dataset_A']['data_dir'] = data_dir
        self.isTrain = False



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
         val_split_ratio=opt.val_split_ratio, max_dataset_size=opt.max_dataset_size, workers= opt.n_workers, is_train=False,
          is_training_data=pa.is_train_data)

    fid_cls = FID(KL.total_dataset, opt.dataset['dataset_A']['data_dir'])

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
        generated_remission.append(vis_dict['fake_B'].cpu().detach())
        for k, v in vis_dict.items():
            test_image_results[k].append(v.cpu().detach().numpy())
            n_pics += v.shape[0]
        tq.update(1)

    test_image_results = {k: np.concatenate(v, axis=0) for k, v in test_image_results.items()}
    fid_score = fid_cls.fid_score(generated_remission)
    losses = {k: np.array(v).mean() for k , v in test_losses.items()}
    print (losses)
    print('FID score: ', fid_score)

    ### save_images


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



    exp_name = os.path.join(opt.checkpoints_dir, opt.name, 'test_results_pics')
    os.makedirs(exp_name, exist_ok=True)
    n_pics = min(n_pics , 100)
    n_keys = len(test_image_results.keys())
    n_pics = n_pics // n_keys
    ra = test_image_results['real_A']
    n_keys = n_keys if ra.shape[1] > 3 else n_keys + 2
    # for i in range(n_pics):
    #     fig = plt.figure()
    #     ind = 0
    #     for k, img in test_image_results.items():
    #         if k == 'real_A' and img.shape[1] > 3:
    #             rgb = img[:, 3:]
    #             ax = fig.add_subplot(2, n_keys // 2, ind+1)
    #             ax.imshow(subsample(rgb[i]).transpose((1, 2, 0)))
    #             ax.title.set_text('rgb')
    #             ax.set_xticks([])
    #             ax.set_yticks([])
    #             img = img[:, :3]
    #             ind += 1
    #             continue

    #         for j in range(img.shape[1]):
    #             ax = fig.add_subplot(2, n_keys//2, ind+1)
    #             ax.imshow(subsample(img[i][j]),
    #                         cmap='inferno' if k == 'range' else 'cividis', vmin=0.0, vmax=1.0)
    #             ax.title.set_text(k)
    #             ax.set_xticks([])
    #             ax.set_yticks([])
    #             ind+= 1
    #     fname = os.path.join(exp_name, 'img_' + str(i) + '.png' )                    
    #     plt.savefig(fname)
    #     plt.close(fig)

    def save_img(img, tag, pic_dir, cmap=None):
        fig = plt.figure()
        if cmap is not None:
            plt.imshow(img, cmap=cmap, vmin=0.0, vmax=1.0)
        else:
            plt.imshow(img)
        plt.axis('off')
        # ax = fig.add_subplot(1, 1, 1)
        # ax.imshow(img)
        # ax.set_xticks([])
        # ax.set_yticks([])
        fname = os.path.join(pic_dir, 'img_' + tag + '.png')
        plt.savefig(fname, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
    for i in range(n_pics):
        pic_dir = os.path.join(exp_name, 'img_' + str(i))
        os.makedirs(pic_dir , exist_ok=True)
        
        ind = 0
        for k, img in test_image_results.items():
            if k == 'real_A' and img.shape[1] > 3:
                rgb = img[:, 3:]
                save_img(subsample(rgb[i]).transpose((1, 2, 0)), 'rgb', pic_dir)
                img = img[:, :3]
                ind += 1
                continue

            # cmap = 'gray' if k == 'range' else 'gray'
            cmap = 'inferno' if k == 'range' else 'cividis'
            for j in range(img.shape[1]):
                save_img(subsample(img[i][j]), k, pic_dir, cmap)
                ind += 1
        
        

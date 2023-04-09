import numpy as np
import os
import sys
import ntpath
import time
import datetime
import shutil
from . import util, html
from subprocess import Popen, PIPE
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from util import colorize, postprocess, flatten
import open3d as o3d
import open3d.visualization.rendering as rendering
import torch
from glob import glob
import yaml

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

def visualize_tensor(pts, depth):

    # depth_range = np.exp2(lidar_range*6)-1
    color = plt.cm.viridis(np.clip(depth, 0, 1).flatten())
    # pts, mask = range_image_to_point_cloud(depth, intensity, H, W)
    # mask out invalid points
    xyz = pts
    color = color[..., :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(color)
    # o3d.visualization.draw_geometries([pcd],zoom=0.25, front=[0.0, 0.0, 1.0],
    #                               lookat=[0.0, 0.0, 0.0],
    #                               up=[1.0, 0.0, 0.0])
    # offscreen rendering
    render = rendering.OffscreenRenderer(1920, 1080, headless=True)
    mtl = rendering.MaterialRecord()
    mtl.base_color = [1, 1, 1, 0.5]
    mtl.point_size = 4
    mtl.shader = "defaultLit"
    render.scene.set_background([255, 255, 255, 0.0])
    render.scene.add_geometry("point cloud", pcd, mtl)
    render.scene.set_lighting(render.scene.LightingProfile.NO_SHADOWS, (1, 1, 1))
    render.scene.scene.enable_sun_light(True)
    render.scene.camera.look_at([0, 0, 0], [0, 0, 1], [0, 1, 0])
    bev_img = render.render_to_image()
    render.setup_camera(60.0, [0, 0, 0], [-0.2, 0, 0.1], [0, 0, 1])
    pts_img = render.render_to_image()
    return bev_img, pts_img

def to_np(tensor):
    return tensor.detach().cpu().numpy()

def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt, lidar, dataset_name='kitti'):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.lidar = lidar
        self.dataset_name = dataset_name
        self.opt = opt
        exp_dir = os.path.join(opt.checkpoints_dir, opt.name)

        self.tb_dir = os.path.join(exp_dir +('/TB/' if opt.isTrain else '/TB_test/'), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tb_dir)
        self.log_name = os.path.join(self.tb_dir , 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def log_imgs(self, tensor, tag, step, color=True, cmap='turbo'):
        B = tensor.shape[0]
        nrow = 4 if B > 8 else 1
        grid = make_grid(tensor.detach(), nrow=nrow)
        grid = grid.cpu().numpy()  # CHW
        if color:
            grid = grid[0]  # HW
            grid = colorize(grid, cmap=cmap).transpose(2, 0, 1)  # CHW
        else:
            grid = grid.astype(np.uint8)
        self.writer.add_image(tag, grid, step)

    def display_current_results(self, phase, visuals, g_step, data_maps):
        visuals = postprocess(visuals, self.lidar, data_maps=data_maps, dataset_name=self.dataset_name)
        for k , v in visuals.items():
            if 'points' in k:
                points = flatten(v)
                inv = visuals[k.replace('points', 'inv')]
                image_list = []
                for i in range(points.shape[0]):
                    _, gen_pts_img = visualize_tensor(to_np(points[i]), to_np(inv[i]) * 2.5)
                    image_list.append(torch.from_numpy(np.asarray(gen_pts_img)))
                visuals[k] = torch.stack(image_list, dim=0).permute(0, 3, 1, 2)
        for k , img_tensor in visuals.items():
            color = False if ('points' in k or 'label' in k) else True
            cmap = 'viridis' if ('reflectance' in k) else 'turbo'
            self.log_imgs(img_tensor, phase + '/' + k, g_step, color, cmap)

    def plot_current_losses(self, phase, epoch, losses, g_step):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        for tag , loss in losses.items():
            self.writer.add_scalar(phase + '/' + tag, loss, g_step)

        # plotting epoch    
        self.writer.add_scalar('epoch', epoch, g_step)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, phase,epoch, iters, losses, tq=None):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = 'Validation\n' if phase == 'val' else ''
        message = message + '(epoch: %d, iters: %d) ' % (epoch, iters)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        if not tq:
            print(message)  # print the message
        else:
            tq.write(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

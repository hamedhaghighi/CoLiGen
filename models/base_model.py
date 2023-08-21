import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks
from util.metrics.cov_mmd_1nna import compute_cd
from util.metrics.depth import compute_depth_accuracy, compute_depth_error
from util.util import SSIM
from util import *

class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.training.gpu_ids
        self.isTrain = opt.training.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.training.checkpoints_dir, opt.training.name)  # save all the checkpoints to save_dir
        # if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
        #     torch.backends.cudnn.benchmark = True
        self.eval_metrics = ['cd', 'depth_errors', 'depth_accuracies']
        self.loss_names = []
        self.extra_val_loss_names = []
        self.model_names = []
        self.visual_names = []
        ds_A_modality = opt.dataset.dataset_A.modality
        self.visual_names.extend(['real_' + m for m in ds_A_modality])
        if hasattr(opt.dataset, 'dataset_B'):
            ds_B_modality = opt.dataset.dataset_B.modality
            self.visual_names.extend(['real_B_' + m for m in ds_B_modality])
        self.visual_names.extend(['real_mask', 'real_B_mask', 'synth_mask'])
        self.visual_names = [s.replace('depth','inv') for s in self.visual_names]
        self.optimizers = []
        self.image_paths = []
        self.crterionSSIM = SSIM()
        self.metric = 0  # used for learning rate policy 'plateau'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    # @abstractmethod
    # def set_input(self, input):
    #     """Unpack input data from the dataloader and perform necessary pre-processing steps.

    #     Parameters:
    #         input (dict): includes the data itself and its metadata information.
    #     """
    #     pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass
    
    @abstractmethod
    def data_dependent_initialize(self):
        pass

    @abstractmethod
    def set_seg_model(self, model):
        pass
    
    def calc_supervised_metrics(self, no_inv, lidar_A, lidar_B):
        self.forward()
        synth_inv = self.real_inv * self.synth_mask if no_inv else self.synth_inv
        points_gen = lidar_B.inv_to_xyz(tanh_to_sigmoid(synth_inv))
        points_gen = flatten(points_gen)
        points_ref = flatten(self.real_points)
        depth_ref = lidar_A.revert_depth(tanh_to_sigmoid(self.real_inv), norm=False)
        depth_gen = lidar_B.revert_depth(tanh_to_sigmoid(synth_inv), norm=False)
        self.cd = compute_cd(points_ref, points_gen).mean().item()
        accuracies = compute_depth_accuracy(depth_ref, depth_gen)
        self.depth_accuracies = {'depth/' + k: v.mean().item() for k ,v in accuracies.items()}
        errors = compute_depth_error(depth_ref, depth_gen)
        self.depth_errors = {'depth/' + k: v.mean().item() for k ,v in errors.items()}
        if 'reflectance' in self.opt.model.modality_B:
            reflectance_ref = tanh_to_sigmoid(self.real_reflectance) + 1e-8
            reflectance_gen = tanh_to_sigmoid(self.synth_reflectance) + 1e-8
            errors = compute_depth_error(reflectance_ref, reflectance_gen)
            self.reflectance_errors = {'reflectance/' + k: v.mean().item() for k ,v in errors.items()}
            self.reflectance_ssim = self.crterionSSIM(self.real_reflectance, self.synth_reflectance, torch.ones_like(self.real_reflectance))

        # combine loss and calculate gradients
    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train or opt.test:
            self.load_networks(opt.epoch)
        self.print_networks(opt.verbose)

    def train(self, flag):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train(flag)

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.training.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self, is_eval=False):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        if is_eval:
            for name in self.eval_metrics:
                value = getattr(self, name)
                if isinstance(value, (int, float)):
                    errors_ret[name] = value
                elif isinstance(value, dict):
                    for k , v in value.items():
                        errors_ret[k] = v
        else:
            for name in self.loss_names:
                errors_ret[name] = float(getattr(self, 'loss_' + name)) 
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        save_dict = dict()
        save_filename = f'{epoch}.pth' if (epoch == 'latest' or epoch == 'best') else f'e_{epoch}.pth'
        save_path = os.path.join(self.save_dir, save_filename)
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    save_dict[name] = net.module.state_dict()
                else:
                    save_dict[name] = net.state_dict()
        for i, o in enumerate(self.optimizers):
            save_dict[f'optimizer_{i}'] = o.state_dict()
        for i, s in enumerate(self.schedulers):
            save_dict[f'scheduler_{i}'] = s.state_dict()
        torch.save(save_dict, save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        load_filename = f'{epoch}.pth' if (epoch == 'latest' or epoch == 'best') else f'e_{epoch}.pth'
        load_path = os.path.join(self.save_dir, load_filename)
        if not os.path.exists(load_path):
            # print(f'cannot find the load path {load_path}')
            raise Exception(f'cannot find the load path {load_path}')
        else:
            state_dict = torch.load(load_path)
            for name in self.model_names:
                if isinstance(name, str):
                    net = getattr(self, 'net' + name)
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.module
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata

                    # # patch InstanceNorm checkpoints prior to 0.4
                    # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    #     self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                    net.load_state_dict(state_dict[name])
            if self.isTrain:
                for i, o in enumerate(self.optimizers):
                    o.load_state_dict(state_dict[f'optimizer_{i}'])
                for i, s in enumerate(self.schedulers):
                    s.load_state_dict(state_dict[f'scheduler_{i}'])
            print('loading the model from %s' % load_path)


    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

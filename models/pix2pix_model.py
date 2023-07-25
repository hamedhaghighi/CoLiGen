import torch
import numpy as np
from .base_model import BaseModel
from . import networks
from util import *


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    def __init__(self, opt, lidar):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'mask_bce']
        
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        
        opt_m = opt.model
        opt_t = opt.training
        self.visual_names = []
        self.eval_metrics = ['cd', 'depth_accuracies', 'depth_errors'] 
        
        if 'inv' in opt_m.modality_B:
            self.visual_names.extend(['synth_inv', 'real_inv', 'synth_inv_orig', 'real_label', 'synth_mask', 'real_mask'])
        if 'reflectance' in opt_m.modality_B:
            self.visual_names.extend(['real_reflectance', 'synth_reflectance'])
            self.eval_metrics.append('reflectance_errors')
        input_nc_G = np.array([m2ch[m] for m in opt_m.modality_A]).sum()
        output_nc_G = np.array([m2ch[m] for m in opt_m.out_ch]).sum()
        input_nc_D = np.array([m2ch[m] for m in opt_m.modality_B]).sum()
        same_kernel_size = opt.dataset.dataset_A.img_prop.width == opt.dataset.dataset_A.img_prop.height
        self.netG = networks.define_G(input_nc_G, output_nc_G, opt_m.ngf, opt_m.netG, opt_m.norm,
                                      not opt_m.no_dropout, opt_m.init_type, opt_m.init_gain, self.gpu_ids, opt_m.out_ch, same_kernel_size=same_kernel_size)

        self.netD = networks.define_D(input_nc_D + input_nc_G, opt_m.ndf, opt_m.netD,
                                        opt_m.n_layers_D, opt_m.norm, opt_m.init_type, opt_m.init_gain, self.gpu_ids)

        # define loss functions
        self.criterionGAN = networks.GANLoss(opt_m.gan_mode).to(self.device)
        self.criterionL1 = torch.nn.L1Loss()
        self.BCEwithLogit = torch.nn.BCEWithLogitsLoss()
        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.lidar = lidar
        self.opt_m = opt_m
        if self.isTrain:
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt_t.lr, betas=(opt_t.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt_t.lr, betas=(opt_t.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

   

    def set_input(self, data):
        data = fetch_reals(data, self.lidar, self.device)
        for k, v in data.items():
            setattr(self, 'real_' + k, v)
        data_list = []
        for m in self.opt.model.modality_A:
            assert m in data
            data_list.append(data[m])
        self.real_A = torch.cat(data_list, dim=1)

        data_list = []
        for m in self.opt.model.modality_B:
            assert m in data
            data_list.append(data[m])
        self.real_B = torch.cat(data_list, dim=1)
        

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        out_dict, self.fake_B  = self.netG(self.real_A) # G(A)
        for k , v in out_dict.items():
            setattr(self, 'synth_' + k , v)
        

    def calc_loss_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

    def calc_loss_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
        self.loss_mask_bce = self.BCEwithLogit(self.synth_mask_logit, self.real_mask)
        
        self.loss_G = self.loss_G_GAN * self.opt.model.lambda_LGAN + self.loss_G_L1 * self.opt.model.lambda_L1 + self.loss_mask_bce * self.opt.model.lambda_mask
        

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.calc_loss_D()
        self.loss_D.backward()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.calc_loss_G()
        self.loss_G.backward()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

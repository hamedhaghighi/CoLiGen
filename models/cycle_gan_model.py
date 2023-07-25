import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from util import *
from . import networks
import random
import math
import sys

class CycleGANModel(BaseModel):
    def __init__(self, opt, lidar):
        BaseModel.__init__(self, opt)
        opt_m = opt.model
        opt_t = opt.training
        self.lidar = lidar
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:
            self.model_names = ['G_A', 'G_B']

        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B']
        if self.opt.model.lambda_idt > 0.0:
            self.loss_names.extend(['idt_A', 'idt_B'])

        self.visual_names = []
        for m in opt_m.modality_A:
            self.visual_names.append('real_' + m)
        for m in opt_m.modality_B:
            self.visual_names.append('synth_' + m)
            self.visual_names.append('real_B_' + m)
        
        input_nc_G_A = np.array([m2ch[m] for m in opt_m.modality_A]).sum()
        input_nc_G_B = np.array([m2ch[m] for m in opt_m.modality_B]).sum()
        output_nc_G = np.array([m2ch[m] for m in opt_m.out_ch]).sum()
        input_nc_D = np.array([m2ch[m] for m in opt_m.modality_B]).sum()
        same_kernel_size = opt.dataset.dataset_A.img_prop.width == opt.dataset.dataset_A.img_prop.height
        
        self.netG_A = networks.define_G(input_nc_G_A, output_nc_G, opt_m.ngf, opt_m.netG, opt_m.norm,
                                      not opt_m.no_dropout, opt_m.init_type, opt_m.init_gain, self.gpu_ids, opt_m.out_ch, same_kernel_size=same_kernel_size)
        self.netG_B = networks.define_G(input_nc_G_B, output_nc_G, opt_m.ngf, opt_m.netG, opt_m.norm,
                                      not opt_m.no_dropout, opt_m.init_type, opt_m.init_gain, self.gpu_ids, opt_m.out_ch, same_kernel_size=same_kernel_size)
        
            
        if self.isTrain:
            self.netD_A = networks.define_D(input_nc_D, opt_m.ndf, opt_m.netD,
                                        opt_m.n_layers_D, opt_m.norm, opt_m.init_type, opt_m.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(input_nc_D, opt_m.ndf, opt_m.netD,
                                        opt_m.n_layers_D, opt_m.norm, opt_m.init_type, opt_m.init_gain, self.gpu_ids)

        if self.isTrain:
            self.old_lr = opt_t.lr
            self.fake_A_pool = ImagePool(opt_t.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt_t.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt_m.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt_t.lr, betas=(opt_t.beta1, 0.999))
            
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt_t.lr, betas=(opt_t.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        data_A = fetch_reals(input['A'], self.lidar, self.device)
        data_B = fetch_reals(input['B'], self.lidar, self.device)
        for k, v in data_A.items():
            setattr(self, 'real_' + k, v)
        for k, v in data_B.items():
            setattr(self, 'real_B_' + k, v)
        self.real_A = cat_modality(data_A, self.opt.model.modality_A)
        self.real_B = cat_modality(data_B, self.opt.model.modality_B)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D


    def forward(self):
        out_dict_B, self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        _, self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        _, self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        _, self.rec_B = self.netG_A(self.fake_A) 
        for k , v in out_dict_B.items():
            setattr(self, 'synth_' + k , v)

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B).item()

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A).item()


    def backward_G(self):
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.opt.model.lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.opt.model.lambda_B
        if self.opt.model.lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            _, self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * self.opt.model.lambda_B * self.opt.model.lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            _, self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * self.opt.model.lambda_A * self.opt.model.lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B

        self.loss_G.backward()


    def optimize_parameters(self):
        # forward
        self.forward()
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad() 
        self.backward_G()
        self.optimizer_G.step() 
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights


 

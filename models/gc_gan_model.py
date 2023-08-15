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

class GcGANModel(BaseModel):
    def __init__(self, opt, lidar_A, lidar_B):
        BaseModel.__init__(self, opt)
        opt_m = opt.model
        opt_t = opt.training
        self.lidar_A = lidar_A
        self.lidar_B = lidar_B
        if self.isTrain:
            self.model_names = ['G_AB', 'G_gc_AB'] if 'cross' in opt_m.name else ['G_AB']
        else:
            self.model_names = ['G_AB', 'G_gc_AB', 'D_B','D_gc_B']

        self.loss_names = ['D_B', 'G_AB', 'G_gc_AB', 'gc']
        if self.opt.model.identity > 0.0:
            self.loss_names.extend(['idt', 'idt_gc'])

        self.visual_names = []
        for m in opt_m.modality_A:
            self.visual_names.append('real_' + m)
        for m in opt_m.modality_B:
            self.visual_names.append('synth_' + m)
            self.visual_names.append('real_B_' + m)
        
        input_nc_G = np.array([m2ch[m] for m in opt_m.modality_A]).sum()
        output_nc_G = np.array([m2ch[m] for m in opt_m.out_ch]).sum()
        input_nc_D = np.array([m2ch[m] for m in opt_m.modality_B]).sum()
        same_kernel_size = opt.dataset.dataset_A.img_prop.width == opt.dataset.dataset_A.img_prop.height
        
        self.netG_AB = networks.define_G(input_nc_G, output_nc_G, opt_m.ngf, opt_m.netG, opt_m.norm,
                                      not opt_m.no_dropout, opt_m.init_type, opt_m.init_gain, self.gpu_ids, opt_m.out_ch, same_kernel_size=same_kernel_size)
        if 'cross' in opt_m.name:
            self.netG_gc_AB = networks.define_G(input_nc_G, output_nc_G, opt_m.ngf, opt_m.netG, opt_m.norm,
                                      not opt_m.no_dropout, opt_m.init_type, opt_m.init_gain, self.gpu_ids, opt_m.out_ch, same_kernel_size=same_kernel_size)
        elif 'share' in opt_m.name:
            self.netG_gc_AB = self.netG_AB
        
            
        if self.isTrain:
            
            self.netD_B = networks.define_D(input_nc_D, opt_m.ndf, opt_m.netD,
                                        opt_m.n_layers_D, opt_m.norm, opt_m.init_type, opt_m.init_gain, self.gpu_ids)
            self.netD_gc_B = networks.define_D(input_nc_D, opt_m.ndf, opt_m.netD,
                                        opt_m.n_layers_D, opt_m.norm, opt_m.init_type, opt_m.init_gain, self.gpu_ids)

        if self.isTrain:
            self.old_lr = opt_t.lr
            self.fake_B_pool = ImagePool(opt_t.pool_size)
            self.fake_gc_B_pool = ImagePool(opt_t.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt_m.gan_mode).to(self.device)
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionGc = torch.nn.L1Loss()
            # initialize optimizers
            if 'cross' in opt_m.name:
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_AB.parameters(), self.netG_gc_AB.parameters()), lr=opt_t.lr, betas=(opt_t.beta1, 0.999))
            elif 'share' in opt_m.name:
                self.optimizer_G = torch.optim.Adam(self.netG_AB.parameters(), lr=opt_t.lr, betas=(opt_t.beta1, 0.999))
            
            self.optimizer_D_B = torch.optim.Adam(itertools.chain(self.netD_B.parameters(), self.netD_gc_B.parameters()), lr=opt_t.lr, betas=(opt_t.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_B)

    def set_seg_model(self, model):
        return
    def data_dependent_initialize(self, data):
        return

    def set_input(self, input):
        data_A = fetch_reals(input['A'], self.lidar_A, self.device)
        data_B = fetch_reals(input['B'], self.lidar_B, self.device)
        for k, v in data_A.items():
            setattr(self, 'real_' + k, v)
        for k, v in data_B.items():
            setattr(self, 'real_B_' + k, v)
        self.real_A = cat_modality(data_A, self.opt.model.modality_A)
        self.real_B = cat_modality(data_B, self.opt.model.modality_B)
        self.real_B_mod_A = cat_modality(data_B, self.opt.model.modality_A)
        self.real_A_mod_B = cat_modality(data_A, self.opt.model.modality_B)

    def backward_D_basic(self, netD, real, fake, netD_gc, real_gc, fake_gc):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        # Real_gc
        pred_real_gc = netD_gc(real_gc)
        loss_D_gc_real = self.criterionGAN(pred_real_gc, True)
        # Fake_gc
        pred_fake_gc = netD_gc(fake_gc.detach())
        loss_D_gc_fake = self.criterionGAN(pred_fake_gc, False)
        # Combined loss
        loss_D += (loss_D_gc_real + loss_D_gc_fake) * 0.5

        # backward
        loss_D.backward()
        return loss_D


    def rot90(self, tensor, direction):
        tensor = tensor.transpose(2, 3)
        size = tensor.shape[3] if direction == 0 else tensor.shape[2]
        inv_idx = torch.arange(size-1, -1, -1).long().cuda()
        if direction == 0:
          tensor = torch.index_select(tensor, 3, inv_idx)
        else:
          tensor = torch.index_select(tensor, 2, inv_idx)
        return tensor

    def forward(self):
        out_dict, self.fake_B = self.netG_AB.forward(self.real_A)
        for k , v in out_dict.items():
            setattr(self, 'synth_' + k , v)

        size = self.real_A.shape[2]

        if self.opt.model.geometry == 'rot':
          self.real_gc_A = self.rot90(self.real_A, 0)
          self.real_gc_B = self.rot90(self.real_B, 0)
          self.real_gc_B_mod_A = self.rot90(self.real_B_mod_A, 0)
        elif self.opt.model.geometry == 'vf':
          inv_idx = torch.arange(size-1, -1, -1).long().cuda()
          self.real_gc_A = torch.index_select(self.real_A, 2, inv_idx)
          self.real_gc_B = torch.index_select(self.real_B, 2, inv_idx)
          self.real_gc_B_mod_A = torch.index_select(self.real_B_mod_A, 2, inv_idx)
        else:
          raise ValueError("Geometry transformation function [%s] not recognized." % self.opt.geometry)

    def get_gc_rot_loss(self, AB, AB_gc, direction):
        loss_gc = 0.0

        if direction == 0:
          AB_gt = self.rot90(AB_gc.clone().detach(), 1)
          loss_gc = self.criterionGc(AB, AB_gt)
          AB_gc_gt = self.rot90(AB.clone().detach(), 0)
          loss_gc += self.criterionGc(AB_gc, AB_gc_gt)
        else:
          AB_gt = self.rot90(AB_gc.clone().detach(), 0)
          loss_gc = self.criterionGc(AB, AB_gt)
          AB_gc_gt = self.rot90(AB.clone().detach(), 1)
          loss_gc += self.criterionGc(AB_gc, AB_gc_gt)

        loss_gc = loss_gc*self.opt.model.lambda_AB*self.opt.model.lambda_gc
        return loss_gc

    def get_gc_vf_loss(self, AB, AB_gc):
        loss_gc = 0.0

        size = AB.shape[2]

        inv_idx = torch.arange(size-1, -1, -1).long().cuda()

        AB_gt = torch.index_select(AB_gc.clone().detach(), 2, inv_idx)
        loss_gc = self.criterionGc(AB, AB_gt)

        AB_gc_gt = torch.index_select(AB.clone().detach(), 2, inv_idx)
        loss_gc += self.criterionGc(AB_gc, AB_gc_gt)

        loss_gc = loss_gc*self.opt.model.lambda_AB*self.opt.model.lambda_gc
        return loss_gc

    def backward_D_B(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        fake_gc_B = self.fake_gc_B_pool.query(self.fake_gc_B)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_B, fake_B, self.netD_gc_B, self.real_gc_B, fake_gc_B)
        self.loss_D_B = loss_D_B.item()

    def backward_G(self):
        # adversariasl loss
        fake_B = self.fake_B
        pred_fake = self.netD_B.forward(fake_B)
        loss_G_AB = self.criterionGAN(pred_fake, True) * self.opt.model.lambda_G
        
        _, fake_gc_B = self.netG_gc_AB.forward(self.real_gc_A)
        pred_fake = self.netD_gc_B.forward(fake_gc_B)
        loss_G_gc_AB = self.criterionGAN(pred_fake, True) * self.opt.model.lambda_G

        if self.opt.model.geometry == 'rot':
            loss_gc = self.get_gc_rot_loss(fake_B, fake_gc_B, 0)
        elif self.opt.model.geometry == 'vf':
            loss_gc = self.get_gc_vf_loss(fake_B, fake_gc_B)

        if self.opt.model.identity > 0:
            # G_AB should be identity if real_B is fed.
            _, idt_A = self.netG_AB(self.real_B_mod_A)
            loss_idt = self.criterionIdt(idt_A, self.real_B) * self.opt.model.lambda_AB * self.opt.model.identity
            _, idt_gc_A = self.netG_gc_AB(self.real_gc_B_mod_A)
            loss_idt_gc = self.criterionIdt(idt_gc_A, self.real_gc_B) * self.opt.model.lambda_AB * self.opt.model.identity

            self.idt_A = idt_A.data
            self.idt_gc_A = idt_gc_A.data
            self.loss_idt = loss_idt.item()
            self.loss_idt_gc = loss_idt_gc.item()
        else:
            loss_idt = 0
            loss_idt_gc = 0
            self.loss_idt = 0
            self.loss_idt_gc = 0

        loss_G = loss_G_AB + loss_G_gc_AB + loss_gc + loss_idt + loss_idt_gc

        loss_G.backward()

        self.fake_B = fake_B.data
        self.fake_gc_B = fake_gc_B.data

        self.loss_G_AB = loss_G_AB.item()
        self.loss_G_gc_AB= loss_G_gc_AB.item()
        self.loss_gc = loss_gc.item()


    def optimize_parameters(self):
        # forward
        self.forward()
        # G_AB
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_B and D_gc_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()


 

import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from .patchnce import PatchNCELoss
from util import *
from . import networks
import random
import math
import sys
from rangenet.tasks.semantic.modules.segmentator import *

class CUTModel(BaseModel):
    def __init__(self, opt, lidar):
        BaseModel.__init__(self, opt)
        opt_m = opt.model
        opt_t = opt.training
        self.lidar = lidar
        if self.isTrain:
            self.model_names = ['G', 'F', 'D', 'F_feat']
        else:
            self.model_names = ['G']
        
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G']
        self.nce_layers = [int(i) for i in opt_m.nce_layers.split(',')]
        if self.opt.model.nce_idt and self.opt.model.lambda_NCE and self.isTrain:
            self.loss_names.extend(['NCE_Y_pix'])
        if self.opt.model.nce_idt and self.opt.model.lambda_NCE_feat and self.isTrain:
            self.loss_names.extend(['NCE_Y_feat'])
        if self.opt.model.lambda_NCE and self.isTrain:
            self.loss_names.extend(['NCE_pix'])
        if self.opt.model.lambda_NCE_feat and self.isTrain:
            self.loss_names.extend(['NCE_feat'])
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
        self.netG = networks.define_G(input_nc_G, output_nc_G, opt_m.ngf, opt_m.netG, opt_m.normG, not opt_m.no_dropout, opt_m.init_type, opt_m.init_gain, self.gpu_ids, opt_m.out_ch, opt_m.no_antialias, opt_m.no_antialias_up, opt=opt_m)
        self.netF = networks.define_F(input_nc_G, opt_m.netF, opt_m.normG, not opt_m.no_dropout, opt_m.init_type, opt_m.init_gain,  self.gpu_ids, opt_m.no_antialias, opt_m)
        self.netF_feat = networks.define_F(input_nc_G, opt_m.netF, opt_m.normG, not opt_m.no_dropout, opt_m.init_type, opt_m.init_gain,  self.gpu_ids, opt_m.no_antialias, opt_m)
        if opt_m.lambda_NCE_feat > 0.0:
            with torch.no_grad():
                self.seg_model = Segmentator().to(self.device)

            
        if self.isTrain:
            self.netD = networks.define_D(input_nc_D, opt_m.ndf, opt_m.netD, opt_m.n_layers_D, opt_m.normD, opt_m.init_type, opt_m.init_gain, self.gpu_ids, opt_m.no_antialias, opt_m)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt_m.gan_mode).to(self.device)
            self.criterionNCE = []
            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt_m, opt_t.batch_size).to(self.device))
            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt_t.lr, betas=(opt_t.beta1, opt_t.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt_t.lr, betas=(opt_t.beta1, opt_t.beta2))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"]['depth'].size(0) // max(len(self.opt.training.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.isTrain:
            self.backward_D()                 # calculate gradients for D
            self.backward_G()                 # calculate graidents for G
            if self.opt.model.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.training.lr, betas=(self.opt.training.beta1, self.opt.training.beta2))
                self.optimizers.append(self.optimizer_F)
            if self.opt.model.lambda_NCE_feat > 0.0:
                self.optimizer_F_feat = torch.optim.Adam(self.netF_feat.parameters(), lr=self.opt.training.lr, betas=(self.opt.training.beta1, self.opt.training.beta2))
                self.optimizers.append(self.optimizer_F_feat)

    def set_input(self, input):
        data_A = fetch_reals(input['A'], self.lidar, self.device)
        data_B = fetch_reals(input['B'], self.lidar, self.device)
        for k, v in data_A.items():
            setattr(self, 'real_' + k, v)
        for k, v in data_B.items():
            setattr(self, 'real_B_' + k, v)
        self.real_A = cat_modality(data_A, self.opt.model.modality_A)
        self.real_B = cat_modality(data_B, self.opt.model.modality_B)
        self.real_B_mod_A = cat_modality(data_B, self.opt.model.modality_A)
        self.real_A_mod_B = cat_modality(data_A, self.opt.model.modality_B)
        self.data_A = data_A
        self.data_B = data_B


    def forward(self):
        self.real = torch.cat((self.real_A, self.real_B_mod_A), dim=0) if self.opt.model.nce_idt and self.isTrain else self.real_A
        if self.opt.model.flip_equivariance:
            self.flipped_for_equivariance = self.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])
        out_dict, self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.model.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]
        for k , v in out_dict.items():
            setattr(self, 'synth_' + k , v[:self.real_A.size(0)])
        for k , v in out_dict.items():
            setattr(self, 'synth_idt_B_' + k , v[self.real_A.size(0):])

    def backward_D(self):
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        # adversariasl loss
        fake_B = self.fake_B

        if self.opt.model.lambda_GAN > 0.0:
            pred_fake = self.netD(fake_B)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.model.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        self.loss_NCE_pix, self.loss_NCE_feat, self.loss_NCE_bd = 0.0, 0.0, 0.0

        if self.opt.model.lambda_NCE > 0.0:
            self.loss_NCE_pix = self.calculate_NCE_loss(self.real_A, self.fake_B)
        if self.opt.model.lambda_NCE_feat > 0.0:
            src_vol = prepare_data_for_seg(self.data_A, self.lidar)
            tgt_vol = prepare_synth_for_seg(self, self.lidar)
            self.loss_NCE_feat = self.calculate_NCE_feat_loss(src_vol, tgt_vol)
            
        loss_NCE_both_pix, loss_NCE_both_feat = self.loss_NCE_pix, self.loss_NCE_feat

        if self.opt.model.nce_idt and self.opt.model.lambda_NCE > 0.0:
            self.loss_NCE_Y_pix = self.calculate_NCE_loss(self.real_B_mod_A, self.idt_B)
            loss_NCE_both_pix = (loss_NCE_both_pix + self.loss_NCE_Y_pix) * 0.5
        if self.opt.model.nce_idt and self.opt.model.lambda_NCE_feat > 0.0:
            src_vol = prepare_data_for_seg(self.data_B, self.lidar)
            tgt_vol = prepare_synth_for_seg(self, self.lidar, 'synth_idt_B')
            self.loss_NCE_Y_feat = self.calculate_NCE_feat_loss(src_vol, tgt_vol)
            loss_NCE_both_feat = (loss_NCE_both_feat + self.loss_NCE_Y_feat) * 0.5

        loss_NCE_both = (loss_NCE_both_pix + loss_NCE_both_feat)
        self.loss_G = self.loss_G_GAN + loss_NCE_both


        self.loss_G.backward()


    def optimize_parameters(self):
        # forward
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.model.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        if self.opt.model.lambda_NCE_feat > 0.0:
            self.optimizer_F_feat.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        if self.opt.model.netF == 'mlp_sample':
            self.optimizer_F.step()
        if self.opt.model.lambda_NCE_feat > 0.0:
            self.optimizer_F_feat.step()

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        if len(self.opt.model.modality_A) > len(self.opt.model.modality_B):
            diff_ch = 0
            for m in set(self.opt.model.modality_A).difference(self.opt.model.modality_B):
                diff_ch += m2ch[m]
            B, C, H ,W = tgt.shape
            extra_ch = torch.zeros(B, diff_ch, H, W).to(src)
            # if src.shape[1] > tgt.shape[1]:
            src[:, 0:diff_ch] = extra_ch  
            tgt = torch.cat([extra_ch, tgt], dim=1)
      
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.model.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.model.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.model.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.model.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers


    def calculate_NCE_feat_loss(self, src_vol, tgt_vol):
        _, feat_q = self.seg_model(tgt_vol)
        _, feat_k = self.seg_model(src_vol)
        feat_q = [feat_q]
        feat_k = [feat_k]
        feat_k_pool, sample_ids = self.netF_feat(feat_k, self.opt.model.num_patches, None)
        feat_q_pool, _ = self.netF_feat(feat_q, self.opt.model.num_patches, sample_ids)
        total_nce_loss = 0.0
        for f_q, f_k, crit, _ in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.model.lambda_NCE_feat
            total_nce_loss += loss.mean()
        return total_nce_loss / len(self.nce_layers)
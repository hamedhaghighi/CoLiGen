import torch
from .base_model import BaseModel
from . import networks
from util.util import SSIM
from util import sigmoid_to_tanh

class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt, lidar):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'mask_bce']
        self.extra_val_loss_names = ['ssim']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['synth_inv', 'real_inv', 'synth_inv_orig', 'real_label', 'synth_mask', 'real_mask']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        opt_m = opt.model
        opt_t = opt.training
        self.netG = networks.define_G(opt_m.input_nc, opt_m.output_nc, opt_m.ngf, opt_m.netG, opt_m.norm,
                                      not opt_m.no_dropout, opt_m.init_type, opt_m.init_gain, self.gpu_ids, opt_m.out_ch)

        self.netD = networks.define_D(opt_m.input_nc + opt_m.output_nc, opt_m.ndf, opt_m.netD,
                                        opt_m.n_layers_D, opt_m.norm, opt_m.init_type, opt_m.init_gain, self.gpu_ids)

        # define loss functions
        self.criterionGAN = networks.GANLoss(opt_m.gan_mode).to(self.device)
        self.criterionL1 = torch.nn.L1Loss()
        self.crterionSSIM = SSIM()
        self.BCEwithLogit = torch.nn.BCEWithLogitsLoss()
        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.lidar = lidar

        if self.isTrain:
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt_t.lr, betas=(opt_m.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt_t.lr, betas=(opt_m.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def fetch_reals(self, data):
        mask = data["mask"].float()
        inv = self.lidar.invert_depth(data["depth"])
        inv = sigmoid_to_tanh(inv)  # [-1,1]
        inv = mask * inv + (1 - mask) * -1
        batch = {'inv': inv, 'mask': mask}
        if 'reflectance' in data:
            reflectance =  data["reflectance"] # [0, 1]
            reflectance = sigmoid_to_tanh(reflectance)
            reflectance = mask * reflectance + (1 - mask) * -1
            batch['reflectance'] = reflectance
        if 'label' in data:
            batch['label'] = data['label']
        for k , v in batch.items():
            batch[k] = v.to(self.device)
        return batch

    def set_input_PCL(self, data):
        data = self.fetch_reals(data)
        for k, v in data.items():
            setattr(self, 'real_' + k, v)
        data_list = []
        for m in self.opt.model.modality_A:
            assert m in data
            data_list.append(data[m])
        
        self.real_A = torch.cat(data_list)
        self.real_B = data[self.opt.model.modality_B]

    def evaluate_model(self):
        self.forward()
        self.calc_loss_D()
        self.calc_loss_G(is_eval=True)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        out = self.netG(self.real_A) # G(A)
        self.fake_B = self.out[self.opt.model.modality_B]
        for k , v in out.items():
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

    def calc_loss_G(self, is_eval=True):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
        self.loss_mask_bce = self.BCEwithLogit(self.mask_logit.squeeze(), self.mask)
        if is_eval:
            self.loss_ssim = self.crterionSSIM(self.real_B, self.fake_B, self.mask)
        # combine loss and calculate gradients
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

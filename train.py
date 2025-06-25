"""
Training script for LIDAR point cloud generation models.

This script implements the main training loop for various generative models including:
- pix2pix: Image-to-image translation with conditional GANs
- cycle_gan: Unpaired image-to-image translation
- gc_gan: Geometry-consistent GANs
- cut: Contrastive unpaired translation

The script supports both single and dual dataset training, with comprehensive
evaluation metrics including FID, segmentation accuracy, and unsupervised metrics.
"""

# Standard library imports
import argparse
import os
import random
import shutil
import time
from collections import defaultdict

# Third-party imports
import numpy as np
import torch
import tqdm
import yaml

# Local imports
from dataset.datahandler import get_data_loader
from fid import FID
# from data import create_dataset
from models import create_model
from rangenet.tasks.semantic.modules.segmentator import *
from util import *
from util.lidar import LiDAR
from util.metrics.cov_mmd_1nna import compute_cov_mmd_1nna
from util.metrics.jsd import compute_jsd
from util.metrics.seg_accuracy import compute_seg_accuracy
from util.metrics.swd import compute_swd
from util.sampling.fps import downsample_point_clouds
from util.visualizer import Visualizer

# Environment setup for compatibility
# Set LD_PRELOAD to ensure proper C++ library loading
os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def cycle(iterable):
    """
    Create an infinite iterator that cycles through the given iterable.
    
    Args:
        iterable: The iterable to cycle through
        
    Yields:
        Elements from the iterable in a repeating cycle
    """
    while True:
        for x in iterable:
            yield x


def inv_to_xyz(inv, lidar, tol=1e-8):
    """
    Convert inverse depth representation to 3D point cloud coordinates.
    
    Args:
        inv: Inverse depth tensor from model output
        lidar: LiDAR configuration object containing projection parameters
        tol: Tolerance for numerical stability in coordinate conversion
        
    Returns:
        torch.Tensor: 3D point cloud coordinates with shape (B, N, 3)
    """
    # Convert from tanh to sigmoid activation and clamp to valid range [0, 1]
    inv = tanh_to_sigmoid(inv).clamp_(0, 1)
    
    # Convert inverse depth to 3D coordinates using LiDAR projection
    xyz = lidar.inv_to_xyz(inv, tol)
    
    # Reshape and transpose to (batch_size, num_points, 3) format
    xyz = xyz.flatten(2).transpose(1, 2)  # (B,N,3)
    
    # Downsample point cloud to 512 points for computational efficiency
    xyz = downsample_point_clouds(xyz, 512)
    return xyz


class M_parser:
    """
    Configuration parser class that loads and manages experiment settings.
    
    This class loads configuration from YAML files and provides easy access
    to all experiment parameters including model, training, and dataset settings.
    """
    
    def __init__(self, cfg_path, data_dir, data_dir_B, load, is_test, batch_size):
        """
        Initialize the configuration parser.
        
        Args:
            cfg_path: Path to the configuration YAML file
            data_dir: Path to dataset A directory (can override config)
            data_dir_B: Path to dataset B directory (can override config)
            load: Experiment name to load (if continuing training)
            is_test: Whether to run in test mode
            batch_size: Batch size (can override config)
        """
        # Load configuration from YAML file
        opt_dict = yaml.safe_load(open(cfg_path, "r"))
        dict_class = make_class_from_dict(opt_dict)
        
        # Copy all attributes from the configuration class to this instance
        # This allows easy access to all config parameters as instance attributes
        members = [
            attr
            for attr in dir(dict_class)
            if not callable(getattr(dict_class, attr)) and not attr.startswith("__")
        ]
        for m in members:
            setattr(self, m, getattr(dict_class, m))
        
        # Override data directories if provided as arguments
        if data_dir != "":
            self.dataset.dataset_A.data_dir = data_dir
        if data_dir_B != "":
            self.dataset.dataset_B.data_dir = data_dir_B
        if batch_size != "":
            self.training.batch_size = int(batch_size)
        
        # Set checkpoint directory if loading experiment
        # Extract checkpoint directory from config path
        if load != "":
            self.training.checkpoints_dir = os.path.sep.join(
                cfg_path.split(os.path.sep)[:-2]
            )
        
        # Set training mode based on test flag
        # When testing, disable training mode
        self.training.test = is_test
        self.model.isTrain = self.training.isTrain = not self.training.test
        self.training.epoch_decay = self.training.n_epochs // 2


def modify_opt_for_fast_test(opt):
    """
    Modify training options for fast testing/debugging.
    
    This function reduces the number of epochs, batch size, and other parameters
    to enable quick testing of the pipeline without full training.
    
    Args:
        opt: Training options object to modify
    """
    opt.n_epochs = 2
    opt.epoch_decay = opt.n_epochs // 2
    opt.display_freq = 1
    opt.print_freq = 1
    opt.save_latest_freq = 1
    opt.max_dataset_size = 10
    opt.batch_size = 2


def check_exp_exists(opt, cfg_args):
    """
    Check if experiment directory exists and handle conflicts.
    
    This function:
    1. Generates experiment name based on model configuration
    2. Checks if experiment directory already exists
    3. Prompts user to delete existing directory or exit
    4. Creates new experiment directory if needed
    
    Args:
        opt: Configuration object containing all experiment settings
        cfg_args: Command line arguments
    """
    cfg_path = cfg_args.cfg
    opt_t = opt.training
    opt_m = opt.model
    opt_d = opt.dataset.dataset_A
    
    # Generate modality strings for experiment naming
    # These help create descriptive experiment names based on model configuration
    modality_A = "_".join(opt_m.modality_A)
    if hasattr(opt_m, "modality_cond"):
        cond_modality = "_".join(opt_m.modality_cond)
    out_ch = "_".join(opt_m.out_ch)
    
    # Set experiment name based on configuration
    if cfg_args.load != "":
        # Use provided experiment name for loading
        opt_t.name = cfg_args.load
    elif cfg_args.fast_test:
        # Use simple name for fast testing
        opt_t.name = "test"
    else:
        # Generate descriptive experiment name based on model type
        if "pix2pix" in opt_m.name:
            opt_t.name = (
                f"pix2pix_modality_A_{modality_A}_out_ch_{out_ch}_L_L1_{opt_m.lambda_L1}"
                + f"_L_GAN_{opt_m.lambda_LGAN}_L_mask_{opt_m.lambda_mask}_w_{opt_d.img_prop.width}_h_{opt_d.img_prop.height}"
                + f"_netG_{opt_m.netG}_netD_{opt_m.netD}_batch_size_{opt_t.batch_size}_finesize_{opt_d.img_prop.finesize}_lr_{opt_t.lr}"
            )
        elif "cycle_gan" in opt_m.name:
            opt_t.name = (
                f"cycle_gan_modality_A_{modality_A}_out_ch_{out_ch}_lambda_A_{opt_m.lambda_A}_lambda_B_{opt_m.lambda_B}_lambda_idt_{opt_m.lambda_idt}"
                + f"_w_{opt_d.img_prop.width}_h_{opt_d.img_prop.height}"
                + f"_netG_{opt_m.netG}_netD_{opt_m.netD}_batch_size_{opt_t.batch_size}_finesize_{opt_d.img_prop.finesize}_lr_{opt_t.lr}"
            )
        elif "gc_gan" in opt_m.name:
            opt_t.name = (
                f"gc_gan_modality_A_{modality_A}_out_ch_{out_ch}_lambda_idt_{opt_m.identity}_lambda_AB_{opt_m.lambda_AB}"
                + f"_lambda_gc_{opt_m.lambda_gc}_lambda_G_{opt_m.lambda_G}_w_{opt_d.img_prop.width}_h_{opt_d.img_prop.height}"
                + f"_netG_{opt_m.netG}_netD_{opt_m.netD}_batch_size_{opt_t.batch_size}_finesize_{opt_d.img_prop.finesize}_lr_{opt_t.lr}"
            )
        elif "cut" in opt_m.name:
            opt_t.name = (
                f"cut_modality_A_{modality_A}_out_ch_{out_ch}_cond_modality_{cond_modality}_lambda_GAN_{opt_m.lambda_GAN}"
                + f"_lambda_NCE_{opt_m.lambda_NCE}_lambda_NCE_feat_{opt_m.lambda_NCE_feat}_w_{opt_d.img_prop.width}_h_{opt_d.img_prop.height}"
                + f"_netG_{opt_m.netG}_netD_{opt_m.netD}_n_layers_D_{opt_m.n_layers_D}_batch_size_{opt_t.batch_size}_finesize_{opt_d.img_prop.finesize}_lr_decay_iters_{opt_t.lr_decay_iters}_lr_{opt_t.lr}"
            )

    # Create experiment directory path
    exp_dir = os.path.join(opt_t.checkpoints_dir, opt_t.name)
    
    # Handle existing experiment directory
    if not opt_t.continue_train and opt_t.isTrain:
        if os.path.exists(exp_dir):
            reply = ""
            # raise Exception('Checkpoint exists!!')
            while not reply.startswith("y") and not reply.startswith("n"):
                reply = (
                    str(
                        input(
                            f"exp_dir {exp_dir} exists. Do you want to delete it? (y/n): \n"
                        )
                    )
                    .lower()
                    .strip()
                )
            if reply.startswith("y"):
                shutil.rmtree(exp_dir)
            else:
                print('Please Re-run the program with "continue train" enabled')
                exit(0)
        os.makedirs(exp_dir, exist_ok=True)
        shutil.copy(cfg_path, exp_dir)
    else:
        assert os.path.exists(exp_dir)


def main(runner_cfg_path=None):
    """
    Main training function for LIDAR point cloud generation models.
    
    This function:
    1. Parses command line arguments and loads configuration
    2. Sets up datasets, models, and evaluation components
    3. Initializes training/validation loops
    4. Performs model training with comprehensive evaluation
    5. Computes and logs various metrics (FID, segmentation, unsupervised)
    
    Args:
        runner_cfg_path: Optional path to configuration file (for programmatic calls)
    """
    # Set up command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="", help="Path of the config file")
    parser.add_argument("--batch_size", type=str, default="", help="batch size")
    parser.add_argument(
        "--data_dir", type=str, default="", help="Path of the dataset A"
    )
    parser.add_argument(
        "--data_dir_B", type=str, default="", help="Path of the dataset B"
    )
    parser.add_argument(
        "--seg_cfg_path", type=str, default="", help="Path of segmentator cfg"
    )
    parser.add_argument(
        "--load",
        type=str,
        default="",
        help="the name of the experiment folder while loading the experiment",
    )
    parser.add_argument("--test", action="store_true", help="test the model")
    parser.add_argument(
        "--map_label",
        action="store_true",
        help="map predicted labels in the case of Semposs dataset",
    )
    parser.add_argument("--norm_label", action="store_true", help="normalise labels")
    parser.add_argument(
        "--fast_test", action="store_true", help="fast test of experiment"
    )
    parser.add_argument(
        "--ref_dataset_name",
        type=str,
        default="",
        help="reference dataset name for measuring unsupervised metrics",
    )
    parser.add_argument(
        "--n_fid", type=int, default=1000, help="num of samples for calculation of fid"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU no")
    parser.add_argument(
        "--on_input",
        action="store_true",
        help="unsupervised metrics will be calculated on dataset A",
    )
    parser.add_argument("--on_real", action="store_true", help="if input is real data")
    parser.add_argument(
        "--no_inv",
        action="store_true",
        help="use it to calc unsupervised metrics on input inv, in case modality_B does not contain inv",
    )
    
    # Parse command line arguments
    cl_args = parser.parse_args()
    torch.cuda.set_device(f"cuda:{cl_args.gpu}")
    if runner_cfg_path is not None:
        cl_args.cfg = runner_cfg_path
    if "checkpoints" in cl_args.cfg:
        cl_args.load = cl_args.cfg.split(os.path.sep)[-2]
    
    # Load configuration and set random seeds for reproducibility
    opt = M_parser(
        cl_args.cfg,
        cl_args.data_dir,
        cl_args.data_dir_B,
        cl_args.load,
        cl_args.test,
        cl_args.batch_size,
    )
    if cl_args.on_real:
        opt.dataset.dataset_A.name = cl_args.ref_dataset_name
    opt.model.norm_label = cl_args.norm_label
    torch.manual_seed(opt.training.seed)
    np.random.seed(opt.training.seed)
    random.seed(opt.training.seed)

    # DATA = yaml.safe_load(open(pa.cfg_dataset, 'r'))
    ## test whole code fast
    if cl_args.fast_test and opt.training.isTrain:
        modify_opt_for_fast_test(opt.training)

    if not opt.training.isTrain:
        opt.training.n_epochs = 1

    # Check experiment directory and set up datasets
    check_exp_exists(opt, cl_args)
    is_two_dataset = False
    if hasattr(opt.dataset, "dataset_B"):
        is_two_dataset = True
    opt.training.gpu_ids = [cl_args.gpu]
    
    # Set up device (GPU or CPU) for computation
    device = (
        torch.device("cuda:{}".format(opt.training.gpu_ids[0]))
        if opt.training.gpu_ids
        else torch.device("cpu")
    )
    
    # Load dataset configurations from YAML files
    ds_cfg = make_class_from_dict(
        yaml.safe_load(
            open(f"configs/dataset_cfg/{opt.dataset.dataset_A.name}_cfg.yml", "r")
        )
    )
    if not hasattr(opt.dataset.dataset_A, "data_dir"):
        opt.dataset.dataset_A.data_dir = ds_cfg.data_dir
    if is_two_dataset:
        if not hasattr(opt.dataset.dataset_B, "data_dir"):
            ds_cfg_B = make_class_from_dict(
                yaml.safe_load(
                    open(
                        f"configs/dataset_cfg/{opt.dataset.dataset_B.name}_cfg.yml", "r"
                    )
                )
            )
            opt.dataset.dataset_B.data_dir = ds_cfg_B.data_dir
    
    # Load reference dataset configuration for metrics computation
    ds_cfg_ref = make_class_from_dict(
        yaml.safe_load(
            open(f"configs/dataset_cfg/{cl_args.ref_dataset_name}_cfg.yml", "r")
        )
    )
    
    # Initialize LiDAR configurations for different datasets
    # These handle the projection and coordinate transformations
    lidar_A = LiDAR(
        cfg=ds_cfg,
        height=opt.dataset.dataset_A.img_prop.height,
        width=opt.dataset.dataset_A.img_prop.width,
    ).to(device)
    lidar_B = (
        LiDAR(
            cfg=ds_cfg_B,
            height=opt.dataset.dataset_B.img_prop.height,
            width=opt.dataset.dataset_B.img_prop.width,
        ).to(device)
        if is_two_dataset
        else None
    )
    lidar_ref = LiDAR(
        cfg=ds_cfg_ref,
        height=opt.dataset.dataset_A.img_prop.height,
        width=opt.dataset.dataset_A.img_prop.width,
    ).to(device)
    lidar = lidar_B if is_two_dataset else lidar_ref
    
    # Create visualizer for output generation
    # This handles saving images, plots, and other visual outputs
    visualizer = Visualizer(
        opt
    )  # create a visualizer that display/save images and plots
    
    # Initialize training variables
    g_steps = 0  # Global step counter
    min_fid = 10000  # Track minimum FID score for best model saving
    
    # Set up ignore labels for different datasets
    # These labels are excluded from segmentation evaluation
    if cl_args.ref_dataset_name == "kitti" or cl_args.ref_dataset_name == "wads":
        ignore_label = [0, 2, 3, 4, 5, 7, 8, 10, 12, 16]
    elif cl_args.ref_dataset_name == "semanticPOSS":
        ignore_label = [0, 3, 9]

    # Set up datasets and data loaders
    is_ref_semposs = cl_args.ref_dataset_name == "semanticPOSS"
    train_dl, train_dataset = get_data_loader(
        opt, "train", opt.training.batch_size, is_ref_semposs=is_ref_semposs
    )
    val_dl, val_dataset = get_data_loader(
        opt,
        "val" if (opt.training.isTrain or cl_args.on_input) else "test",
        opt.training.batch_size,
        shuffle=False,
        is_ref_semposs=is_ref_semposs,
    )
    test_dl, test_dataset = get_data_loader(
        opt,
        "test",
        opt.training.batch_size,
        dataset_name=cl_args.ref_dataset_name,
        two_dataset_enabled=False,
        is_ref_semposs=is_ref_semposs,
    )
    
    # Load segmentation model for evaluation
    # This model performs semantic segmentation on the generated data
    with torch.no_grad():
        seg_model = Segmentator(
            dataset_name=cl_args.ref_dataset_name, cfg_path=cl_args.seg_cfg_path
        ).to(device)
    
    # Create and initialize main model
    model = create_model(
        opt, lidar_A, lidar_B
    )  # create a model given opt.model and other options
    model.set_seg_model(
        seg_model
    )  # regular setup: load and print networks; create schedulers
    ## initilisation of the model for netF in cut
    train_dl_iter = iter(train_dl)
    data = next(train_dl_iter)
    model.data_dependent_initialize(data)
    model.setup(opt.training)
    
    # Initialize FID calculator for evaluation
    fid_cls = (
        FID(seg_model, train_dataset, cl_args.ref_dataset_name, lidar_A)
        if cl_args.ref_dataset_name != ""
        else None
    )
    
    # Prepare real data samples for unsupervised metrics
    n_test_batch = 2 if cl_args.fast_test else len(test_dl)
    test_dl_iter = iter(test_dl)
    data_dict = defaultdict(list)
    N = (
        2 * opt.training.batch_size
        if cl_args.fast_test
        else min(len(test_dataset), len(val_dataset), 1000)
    )
    test_tq = tqdm.tqdm(total=n_test_batch, desc="real_data", position=5)
    for i in range(0, N, opt.training.batch_size):
        data = next(test_dl_iter)
        data = fetch_reals(data, lidar_ref, device, opt.model.norm_label)
        data_dict["real-2d"].append(data["inv"])
        data_dict["real-3d"].append(inv_to_xyz(data["inv"], lidar_ref))
        test_tq.update(1)
    
    # Initialize training progress tracking
    epoch_tq = tqdm.tqdm(total=opt.training.n_epochs, desc="Epoch", position=1)
    start_from_epoch = (
        model.schedulers[0].last_epoch if opt.training.continue_train else 0
    )

    #### Train & Validation Loop
    for epoch in range(
        start_from_epoch, opt.training.n_epochs
    ):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        e_steps = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        
        # Train loop
        if opt.training.isTrain:
            model.train(True)
            train_dl_iter = iter(train_dl)
            n_train_batch = 2 if cl_args.fast_test else len(train_dl)
            train_tq = tqdm.tqdm(total=n_train_batch, desc="Iter", position=3)
            for i in range(n_train_batch):  # inner loop within one epoch
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
                # if epoch == start_from_epoch and i == 0:
                #     model.data_dependent_initialize(data)
                model.set_input(
                    data
                )  # unpack data from dataset and apply preprocessing
                model.optimize_parameters()  # calculate loss functions, get gradients, update network weights
                if (
                    g_steps % opt.training.display_freq == 0
                ):  # display images on visdom and save images to a HTML file
                    current_visuals = model.get_current_visuals()
                    if is_two_dataset:
                        visualizer.display_current_results(
                            "train",
                            current_visuals,
                            g_steps,
                            ds_cfg,
                            opt.dataset.dataset_A.name,
                            lidar_A,
                            ds_cfg_B,
                            opt.dataset.dataset_B.name,
                            lidar_B,
                        )
                    else:
                        visualizer.display_current_results(
                            "train",
                            current_visuals,
                            g_steps,
                            ds_cfg,
                            opt.dataset.dataset_A.name,
                            lidar_A,
                        )

                if (
                    g_steps % opt.training.print_freq == 0
                ):  # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    visualizer.print_current_losses(
                        "train", epoch, e_steps, losses, train_tq
                    )
                    visualizer.plot_current_losses("train", epoch, losses, g_steps)

                if (
                    g_steps % opt.training.save_latest_freq == 0
                ):  # cache our latest model every <save_latest_freq> iterations
                    train_tq.write(
                        "saving the latest model (epoch %d, total_iters %d)"
                        % (epoch, g_steps)
                    )
                    model.save_networks("latest")
                train_tq.update(1)
        
        # Validation loop
        val_dl_iter = iter(val_dl)
        n_val_batch = 2 if cl_args.fast_test else len(val_dl)
        ##### validation
        val_losses = defaultdict(list)
        model.train(False)  # Set model to evaluation mode
        tag = "val" if opt.training.isTrain else "test"
        val_tq = tqdm.tqdm(total=n_val_batch, desc="val_Iter", position=5)
        dis_batch_ind = np.random.randint(0, n_val_batch)  # Random batch for visualization
        data_dict["synth-2d"] = []
        data_dict["synth-3d"] = []
        fid_samples = [] if fid_cls is not None else None
        iou_list = []
        m_acc_list = []
        rec_list = []
        prec_list = []
        label_map = (
            ds_cfg_ref.kitti_to_POSS_map
            if is_ref_semposs and cl_args.map_label
            else None
        )
        
        # Process validation batches
        for i in range(n_val_batch):
            data = next(val_dl_iter)
            model.set_input(data)
            with torch.no_grad():
                model.calc_supervised_metrics(cl_args.no_inv, lidar_A, lidar)

            # Fetch real data for comparison and processing
            fetched_data = fetch_reals(
                data["A"] if is_two_dataset else data, lidar_A, device
            )
            
            # Extract synthetic data based on configuration
            if cl_args.on_input:
                # Use input data as synthetic data (for baseline comparison)
                if "inv" in fetched_data:
                    synth_inv = fetched_data["inv"]
                if "reflectance" in fetched_data:
                    synth_reflectance = fetched_data["reflectance"]
                if "mask" in fetched_data:
                    synth_mask = fetched_data["mask"]
            else:
                # Use model-generated synthetic data
                if hasattr(model, "synth_reflectance"):
                    synth_reflectance = model.synth_reflectance
                if hasattr(model, "synth_mask"):
                    synth_mask = model.synth_mask
                if hasattr(model, "synth_inv") and (
                    not cl_args.no_inv or cl_args.ref_dataset_name == "wads"
                ):
                    synth_inv = model.synth_inv
                else:
                    synth_inv = fetched_data["inv"] * synth_mask
            
            # Special handling for WADS dataset snow generation
            # For snow I add only the generated snow
            if (
                cl_args.ref_dataset_name == "wads"
                and not opt.training.isTrain
                and not cl_args.on_input
                and not cl_args.on_real
                and cl_args.no_inv
            ):
                # Process synthetic data for snow detection
                s_depth = lidar.revert_depth(tanh_to_sigmoid(synth_inv), norm=False)
                s_point = lidar.inv_to_xyz(tanh_to_sigmoid(synth_inv)) * lidar.max_depth
                s_reflectance = tanh_to_sigmoid(synth_reflectance)
                s_data = torch.cat([s_depth, s_point, s_reflectance, synth_mask], dim=1)
                
                # Use segmentation model to detect snow regions
                pred, _ = seg_model(s_data * fetched_data["mask"])
                pred = pred.argmax(dim=1, keepdim=True)
                is_snow = (pred == 20).float()
                
                # Only apply synthetic snow to detected snow regions
                synth_reflectance = (
                    is_snow * synth_reflectance
                    + (1 - is_snow) * fetched_data["reflectance"]
                ) * synth_mask
                synth_inv = (
                    is_snow * synth_inv + (1 - is_snow) * fetched_data["inv"]
                ) * synth_mask
                model.synth_inv = synth_inv
                model.synth_reflectance = synth_reflectance

            # Store synthetic data for unsupervised metrics
            data_dict["synth-2d"].append(synth_inv)
            data_dict["synth-3d"].append(inv_to_xyz(synth_inv, lidar))

            # Prepare FID samples and compute segmentation metrics
            if fid_cls is not None and len(fid_samples) < cl_args.n_fid:
                # Process synthetic data for FID calculation
                synth_depth = lidar.revert_depth(tanh_to_sigmoid(synth_inv), norm=False)
                synth_points = (
                    lidar.inv_to_xyz(tanh_to_sigmoid(synth_inv)) * lidar.max_depth
                )
                synth_reflectance = tanh_to_sigmoid(synth_reflectance)
                synth_data = torch.cat(
                    [synth_depth, synth_points, synth_reflectance, synth_mask], dim=1
                )
                fid_samples.append(synth_data)
                
                # Compute segmentation accuracy if not training
                if not opt.training.isTrain:
                    iou, m_acc, prec, rec = compute_seg_accuracy(
                        seg_model,
                        synth_data * fetched_data["mask"],
                        fetched_data["lwo"],
                        ignore=ignore_label,
                        label_map=label_map,
                    )
                    iou_list.append(iou.cpu().numpy())
                    m_acc_list.append(m_acc.cpu().numpy())
                    prec_list.append(prec.cpu().numpy())
                    rec_list.append(rec.cpu().numpy())

            # Display results for a random batch
            if i == dis_batch_ind:
                current_visuals = model.get_current_visuals()
                if is_two_dataset:
                    visualizer.display_current_results(
                        tag,
                        current_visuals,
                        g_steps,
                        ds_cfg,
                        opt.dataset.dataset_A.name,
                        lidar_A,
                        ds_cfg_B,
                        opt.dataset.dataset_B.name,
                        (
                            lidar_A
                            if cl_args.ref_dataset_name == "wads" and cl_args.no_inv
                            else lidar_B
                        ),
                    )
                else:
                    visualizer.display_current_results(
                        tag,
                        current_visuals,
                        g_steps,
                        ds_cfg,
                        opt.dataset.dataset_A.name,
                        lidar_A,
                    )

            # Collect validation losses
            for k, v in model.get_current_losses(is_eval=True).items():
                val_losses[k].append(v)
            val_tq.update(1)
        
        # Print segmentation results if not training
        if not opt.training.isTrain:
            avg_m_acc = np.array(m_acc_list).mean()
            iou_avg = np.array(iou_list).mean(axis=0)
            prec_avg = np.array(prec_list).mean(axis=0)
            rec_avg = np.array(rec_list).mean(axis=0)
            label_names = seg_model.learning_class_to_label_name(
                np.arange(len(iou_avg)), ds_cfg_ref
            )
            print("avg seg acc:", np.round(avg_m_acc, 2))
            print("iou avg:")
            print_str = ""
            cross_class_iou_avg = iou_avg[iou_avg != 0.0].mean()
            for i, (l, iou) in enumerate(zip(label_names, iou_avg)):
                if iou > 0.0:
                    # print_str = print_str + f'{l}:{np.round(iou, 4)} precision:{np.round(prec_avg[i],4)}, recall:{np.round(rec_avg[i],4)} '
                    print_str = print_str + f"{l}:{np.round(iou, 4)} "
            print(print_str)
            print("cross-class iou:", np.round(cross_class_iou_avg, 2))
        
        # Calculate and log validation losses
        losses = {k: float(np.array(v).mean()) for k, v in val_losses.items()}
        visualizer.plot_current_losses(tag, epoch, losses, g_steps)
        visualizer.print_current_losses(tag, epoch, e_steps, losses, val_tq)

        ##### calculating unsupervised metrics

        # Prepare data for unsupervised metrics calculation
        for k, v in data_dict.items():
            if isinstance(v, list):
                data_dict[k] = torch.cat(v, dim=0)[:N]
        
        # Compute various unsupervised metrics
        scores = {}
        scores.update(compute_swd(data_dict["synth-2d"], data_dict["real-2d"]))
        scores["jsd"] = compute_jsd(
            data_dict["synth-3d"] / 2.0, data_dict["real-3d"] / 2.0
        )
        scores.update(
            compute_cov_mmd_1nna(
                data_dict["synth-3d"], data_dict["real-3d"], 512, ("cd",)
            )
        )
        torch.cuda.empty_cache()
        
        # Compute FID score if FID calculator is available
        if fid_cls is not None:
            scores["fid"] = fid_cls.fid_score(torch.cat(fid_samples, dim=0))
        
        # Save best model based on FID score
        if scores["fid"] < min_fid and opt.training.isTrain:
            min_fid = scores["fid"]
            model.save_networks("best")
        
        # Log unsupervised metrics
        visualizer.plot_current_losses("unsupervised_metrics", epoch, scores, g_steps)
        visualizer.print_current_losses(
            "unsupervised_metrics", epoch, e_steps, scores, val_tq
        )
        
        # Update learning rate if training
        if opt.training.isTrain:
            model.update_learning_rate()  # update learning rates in the beginning of every epoch.
        epoch_tq.update(1)
        print(
            "End of epoch %d \t Time Taken: %d sec"
            % (epoch, time.time() - epoch_start_time)
        )


if __name__ == "__main__":
    main()

"""
Model testing and evaluation script.

This script loads a trained model and evaluates it on test data, computing
various metrics including FID scores and generating visualizations of the results.
It supports both training and test data evaluation modes.
"""

# Standard library imports
import argparse
import os
import time
from collections import defaultdict

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import yaml
from matplotlib import cm
from tqdm import trange

# Local imports
from dataset.datahandler import Loader
# from data import create_dataset
from models import create_model
from util.fid import FID
from util.visualizer import Visualizer


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


class M_parser:
    """
    Configuration parser class for testing.

    This class loads configuration from YAML files and provides easy access
    to all test parameters including model and dataset settings.
    """

    def __init__(self, cfg_path, data_dir):
        """
        Initialize the configuration parser.

        Args:
            cfg_path: Path to the configuration YAML file
            data_dir: Path to dataset directory
        """
        # Load configuration from YAML file
        opt_dict = yaml.safe_load(open(cfg_path, "r"))

        # Copy all attributes from the configuration dictionary to this instance
        for k, v in opt_dict.items():
            setattr(self, k, v)

        # Override data directory if provided
        if data_dir != "":
            self.dataset["dataset_A"]["data_dir"] = data_dir

        # Set training mode to False for testing
        self.isTrain = False


if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_test", type=str, help="Path of the config file")
    parser.add_argument("--data_dir", type=str, default="", help="Path of the dataset")
    parser.add_argument(
        "--is_train_data", "-it", action="store_true", help="is train data"
    )

    # Parse command line arguments
    pa = parser.parse_args()
    opt = M_parser(pa.cfg_test, pa.data_dir)

    # Set random seeds for reproducibility
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    # DATA = yaml.safe_load(open(pa.cfg_dataset, 'r'))
    ## test whole code fast

    # Create and set up model
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    # Create visualizer for result generation
    visualizer = Visualizer(
        opt
    )  # create a visualizer that display/save images and plots

    g_steps = 0

    # Create data loader for testing
    KL = Loader(
        data_dict=opt.dataset,
        batch_size=opt.batch_size,
        val_split_ratio=opt.val_split_ratio,
        max_dataset_size=opt.max_dataset_size,
        workers=opt.n_workers,
        is_train=False,
        is_training_data=pa.is_train_data,
    )

    # Initialize FID calculator for evaluation
    fid_cls = FID(KL.total_dataset, opt.dataset["dataset_A"]["data_dir"])

    e_steps = (
        0  # the number of training iterations in current epoch, reset to 0 every epoch
    )
    visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch

    # Set up test data iterator
    test_dl = iter(KL.testloader)
    n_test_batch = len(KL.testloader)

    # Initialize storage for test results
    test_losses = defaultdict(list)
    test_image_results = defaultdict(list)
    model.train(False)  # Set model to evaluation mode

    # Main testing loop
    tq = tqdm.tqdm(total=n_test_batch, desc="val_Iter", position=5)
    n_pics = 0
    generated_remission = []

    for i in range(n_test_batch):
        # Load test data
        data = next(test_dl)
        model.set_input_PCL(data)

        # Run model evaluation
        with torch.no_grad():
            model.evaluate_model()

        # Collect losses
        for k, v in model.get_current_losses(is_eval=True).items():
            test_losses[k].append(v)

        # Collect visualization results
        vis_dict = model.get_current_visuals()
        generated_remission.append(vis_dict["fake_B"].cpu().detach())
        for k, v in vis_dict.items():
            test_image_results[k].append(v.cpu().detach().numpy())
            n_pics += v.shape[0]
        tq.update(1)

    # Concatenate results from all batches
    test_image_results = {
        k: np.concatenate(v, axis=0) for k, v in test_image_results.items()
    }

    # Compute FID score and average losses
    fid_score = fid_cls.fid_score(generated_remission)
    losses = {k: np.array(v).mean() for k, v in test_losses.items()}
    print(losses)
    print("FID score: ", fid_score)

    ### save_images

    def subsample(img):
        """
        Subsample image by taking every 4th row and normalize to [0, 1] range.

        Args:
            img: Input image tensor

        Returns:
            numpy.ndarray: Subsampled and normalized image
        """
        # img shape C, H , W
        if len(img.shape) == 3:
            _, H, _ = img.shape
        elif len(img.shape) == 2:
            H, _ = img.shape
        y_ind = np.arange(0, H, 4)
        if len(img.shape) == 3:
            return img[:, y_ind, :] * 0.5 + 0.5
        return img[y_ind, :] * 0.5 + 0.5

    # Create output directory for test results
    exp_name = os.path.join(opt.checkpoints_dir, opt.name, "test_results_pics")
    os.makedirs(exp_name, exist_ok=True)
    n_pics = min(n_pics, 100)  # Limit number of images to save
    n_keys = len(test_image_results.keys())
    n_pics = n_pics // n_keys
    ra = test_image_results["real_A"]
    n_keys = n_keys if ra.shape[1] > 3 else n_keys + 2

    # Commented out code for creating combined visualization plots
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
        """
        Save a single image with specified parameters.

        Args:
            img: Image to save
            tag: Tag for the image filename
            pic_dir: Directory to save the image
            cmap: Colormap for the image (optional)
        """
        fig = plt.figure()
        if cmap is not None:
            plt.imshow(img, cmap=cmap, vmin=0.0, vmax=1.0)
        else:
            plt.imshow(img)
        plt.axis("off")
        # ax = fig.add_subplot(1, 1, 1)
        # ax.imshow(img)
        # ax.set_xticks([])
        # ax.set_yticks([])
        fname = os.path.join(pic_dir, "img_" + tag + ".png")
        plt.savefig(fname, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    # Save individual images for each test sample
    for i in range(n_pics):
        pic_dir = os.path.join(exp_name, "img_" + str(i))
        os.makedirs(pic_dir, exist_ok=True)

        ind = 0
        for k, img in test_image_results.items():
            # Handle RGB images separately
            if k == "real_A" and img.shape[1] > 3:
                rgb = img[:, 3:]
                save_img(subsample(rgb[i]).transpose((1, 2, 0)), "rgb", pic_dir)
                img = img[:, :3]
                ind += 1
                continue

            # Save each channel of the image with appropriate colormap
            # cmap = 'gray' if k == 'range' else 'gray'
            cmap = "inferno" if k == "range" else "cividis"
            for j in range(img.shape[1]):
                save_img(subsample(img[i][j]), k, pic_dir, cmap)
                ind += 1

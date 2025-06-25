#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

"""
FID (Fréchet Inception Distance) implementation for LIDAR point cloud evaluation.

This module implements FID score calculation for evaluating the quality of generated
LIDAR point clouds by comparing their feature distributions with real data.
The FID score measures the distance between two multivariate Gaussian distributions
representing the feature activations of real and generated samples.
"""

# Standard library imports
import os
import pickle
import random

# Third-party imports
import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm, trange

# Local imports
from rangenet.tasks.semantic.modules.segmentator import *
from util import _map, prepare_data_for_seg


class FID:
    """
    Fréchet Inception Distance (FID) calculator for LIDAR point cloud evaluation.

    This class computes FID scores by comparing feature distributions of real and
    generated LIDAR data using a pre-trained RangeNet model as the feature extractor.
    """

    def __init__(
        self, model, train_dataset, dataset_name, lidar, max_sample=1000, batch_size=8
    ):
        """
        Initialize the FID calculator.

        Args:
            model: Pre-trained RangeNet model for feature extraction
            train_dataset: Dataset containing real LIDAR samples
            dataset_name: Name of the dataset for caching statistics
            lidar: LIDAR configuration parameters
            max_sample: Maximum number of samples to use for statistics (default: 1000)
            batch_size: Batch size for feature extraction (default: 8)
        """
        self.path = "./"
        self.batch_size = batch_size
        ds = train_dataset
        n_samples = min(max_sample, len(train_dataset))
        stat_dir = os.path.join("fid_stats", f"fid_{dataset_name}.pkl")

        # Store parameters
        self.lidar = lidar
        # concatenate the encoder and the head
        self.model = model

        # use knn post processing?
        # self.post = None
        # if self.ARCH["post"]["KNN"]["use"]:
        #   self.post = KNN(self.ARCH["post"]["KNN"]["params"],
        #                    self.n_classes)

        # Set device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load or compute training statistics
        if os.path.isfile(stat_dir):
            # Load pre-computed statistics from file
            stat = pickle.load(open(stat_dir, "rb"))
            self.mu_train, self.sigma_train = stat["mu"], stat["sigma"]
            print("FID stats loaded ...\n")
        else:
            # Compute statistics from training data
            sample_indxs = np.random.choice(
                range(len(train_dataset)), n_samples, replace=False
            )
            samples = []
            for ind in tqdm(sample_indxs, desc="gathering real samples for fid"):
                data = ds[ind]
                if "B" in data:
                    data = data["B"]
                vol = prepare_data_for_seg(data, lidar, is_batch=False)
                vol = vol.to(self.device)
                samples.append(vol)
            samples = torch.stack(samples, dim=0)
            self.mu_train, self.sigma_train = self.compute_stats(samples)

            # Save statistics for future use
            pickle.dump(
                {"mu": self.mu_train, "sigma": self.sigma_train}, open(stat_dir, "wb")
            )
            print("FID stats saved ...\n")

    def compute_stats(self, data_tensor):
        """
        Compute mean and covariance statistics from feature activations.

        Args:
            data_tensor: Tensor containing LIDAR data samples

        Returns:
            tuple: (mean_vector, covariance_matrix) of feature activations
        """
        # Extract features using the RangeNet model
        feature_array = self.compute_range_net_features(data_tensor)
        _, C, H, W = feature_array.shape

        # Set random seed for reproducible sampling
        random.seed(0)
        # indices = range(4096)

        # Sample 4096 random indices from the feature space
        indices = random.sample(range(0, C * H * W), 4096)
        all_activations = []

        # Extract activations at sampled indices for each sample
        for f in feature_array:
            all_activations.append(f.reshape((-1))[indices])
        all_activations = np.stack(all_activations, axis=0)

        # Compute mean and covariance
        mu = np.mean(all_activations, axis=0)
        sigma = np.cov(all_activations, rowvar=False)
        return mu, sigma

    def compute_range_net_features(self, data_tensor):
        """
        Extract features from LIDAR data using the RangeNet model.

        Args:
            data_tensor: Tensor containing LIDAR data samples

        Returns:
            numpy.ndarray: Feature activations from the model
        """
        # Calculate number of batches needed
        n_batch = np.ceil(len(data_tensor) / self.batch_size)
        features_list = []

        # Process data in batches
        for i in trange(int(n_batch), desc="extracting features for fid"):
            data = data_tensor[i * self.batch_size : (i + 1) * self.batch_size]
            _, feature = self.model(data)
            # a = out[0].argmax(dim=0).detach().cpu().numpy();import matplotlib.pyplot as plt
            # plt.imshow(_map(_map(a, self.DATA['learning_map_inv']), self.DATA['color_map'])[..., ::-1]);plt.show()
            features_list.append(feature.detach().cpu().numpy())

        return np.concatenate(features_list, axis=0)

    def fid_score(self, samples):
        """
        Calculate FID score between generated samples and training data.

        Args:
            samples: Generated LIDAR samples to evaluate

        Returns:
            float: FID score (lower is better)
        """
        # list of tensors in cpu

        assert samples.shape[0] > 1, "for FID num of samples must be greater than one"
        # batch_size = min(batch_size, samples.shape[0])

        # Compute statistics for generated samples
        mu, sigma = self.compute_stats(samples)

        # Calculate Fréchet distance between real and generated distributions
        fid = self.calculate_frechet_distance(
            self.mu_train, self.sigma_train, mu, sigma
        )
        return fid

        # proj_argmax.tofile(path)

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Calculate the Fréchet Distance between two multivariate Gaussian distributions.

        The Fréchet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is:
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))

        This is a stable implementation by Dougal J. Sutherland.

        Args:
            mu1: Mean vector of the first distribution (generated samples)
            sigma1: Covariance matrix of the first distribution (generated samples)
            mu2: Mean vector of the second distribution (real samples)
            sigma2: Covariance matrix of the second distribution (real samples)
            eps: Small value to add to diagonal for numerical stability

        Returns:
            float: The Fréchet Distance between the two distributions
        """
        # Ensure inputs are numpy arrays with correct dimensions
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        # Validate input dimensions
        assert (
            mu1.shape == mu2.shape
        ), "Training and test mean vectors have different lengths"
        assert (
            sigma1.shape == sigma2.shape
        ), "Training and test covariances have different dimensions"

        # Calculate difference between means
        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates"
            ) % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        # Calculate trace of the covariance mean
        tr_covmean = np.trace(covmean)

        # Return the Fréchet distance
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

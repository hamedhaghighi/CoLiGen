import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import inception_v3
import cv2
import multiprocessing
import numpy as np
import glob
import os
from scipy import linalg
import pickle

def to_cuda(elements):
    """
    Transfers elements to cuda if GPU is available
    Args:
        elements: torch.tensor or torch.nn.module
        --
    Returns:
        elements: same as input on GPU memory, if available
    """
    if torch.cuda.is_available():
        return elements.cuda()
    return elements


class PartialInceptionNetwork(nn.Module):

    def __init__(self, transform_input=True):
        super().__init__()
        self.inception_network = inception_v3(pretrained=True)
        self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)
        self.transform_input = transform_input

    def output_hook(self, module, input, output):
        # N x 2048 x 8 x 8
        self.mixed_7c_output = output

    def forward(self, x):
        """
        Args:
            x: shape (N, 3, 299, 299) dtype: torch.float32 in range 0-1
        Returns:
            inception activations: torch.tensor, shape: (N, 2048), dtype: torch.float32
        """
        assert x.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +\
                                             ", but got {}".format(x.shape)
        x = x * 2 - 1  # Normalize to [-1, 1]

        # Trigger output hook
        self.inception_network(x)

        # Output: N x 2048 x 1 x 1
        activations = self.mixed_7c_output
        activations = torch.nn.functional.adaptive_avg_pool2d(
            activations, (1, 1))
        activations = activations.view(x.shape[0], 2048)
        return activations




class FID:

    def __init__(self, train_dataset, data_dir ='', max_sample =10000, batch_size=64):

        ds = train_dataset
        n_samples = min(max_sample, len(train_dataset))
        self.batch_size = min(batch_size, n_samples)
        sample_indxs = np.random.choice(range(len(train_dataset)), n_samples, replace=False)
        samples = []
        for ind in sample_indxs:
            _, _, proj_remission, _, _ = ds[ind]
            samples.append(proj_remission)
        samples = self.preprocess_samples(samples)
        stat_dir = os.path.join(data_dir, 'fid_train_stat_with_net.pkl')
        if os.path.isfile(stat_dir):
            stat = pickle.load(open(stat_dir, 'rb'))
            self.mu_train, self.sigma_train = stat['mu'], stat['sigma']
            inception_network = stat['net']
            inception_network = to_cuda(inception_network)
            inception_network.eval()
            self.inception_network = inception_network
        else:
            inception_network = PartialInceptionNetwork()
            inception_network = to_cuda(inception_network)
            inception_network.eval()
            self.inception_network = inception_network
            self.mu_train , self.sigma_train = self.calculate_activation_statistics(samples, batch_size)
            pickle.dump({'mu': self.mu_train, 'sigma':self.sigma_train, 'net':inception_network }, open(stat_dir, 'wb'))

        

    def fid_score(self, samples):
        # list of tensors in cpu
        samples = self.preprocess_samples(samples)

        assert samples.shape[0] > 1 , 'for FID num of samples must be greater than one'
        batch_size = min(self.batch_size, samples.shape[0])
        mu , sigma = self.calculate_activation_statistics(samples, batch_size)
        fid = self.calculate_frechet_distance(self.mu_train, self.sigma_train, mu , sigma)

        return fid

    def calculate_activation_statistics(self, images, batch_size):
        """Calculates the statistics used by FID
        Args:
            images: torch.tensor, shape: (N, 3, H, W), dtype: torch.float32 in range 0 - 1
            batch_size: batch size to use to calculate inception scores
        Returns:
            mu:     mean over all activations from the last pool layer of the inception model
            sigma:  covariance matrix over all activations from the last pool layer 
                    of the inception model.

        """
        act = self.get_activations(images, batch_size)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma


    def preprocess_samples(self, samples, use_multiprocessing=False):
        """Resizes and shifts the dynamic range of image to 0-1
        Args:
            images: np.array, shape: (N, H, W, 3), dtype: float32 between 0-1 or np.uint8
            use_multiprocessing: If multiprocessing should be used to pre-process the images
        Return:
            final_images: torch.tensor, shape: (N, 3, 299, 299), dtype: torch.float32 between 0-1
        """
        # list of tensors in cpu
        if len(samples[0].shape) > 3:
            samples = torch.cat(samples, dim=0)
        else:
            samples = torch.stack(samples, dim=0)


        B, _, H, W = samples.shape
        samples = samples.expand(B, 3, H, W)
        samples = F.interpolate(samples, (299, 299))
        samples = (samples * 0.5) + 0.5
        assert samples.max() <= 1.0
        assert samples.min() >= 0.0
        assert samples.dtype == torch.float32
        assert samples.shape == (B, 3, 299, 299)
        return samples


    def get_activations(self, images, batch_size):
        """
        Calculates activations for last pool layer for all iamges
        --
            Images: torch.array shape: (N, 3, 299, 299), dtype: torch.float32
            batch size: batch size used for inception network
        --
        Returns: np array shape: (N, 2048), dtype: np.float32
        """
        assert images.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +\
                                                ", but got {}".format(
                                                    images.shape)

        num_images = images.shape[0]
        n_batches = int(np.ceil(num_images / batch_size))
        inception_activations = np.zeros((num_images, 2048), dtype=np.float32)

        for batch_idx in range(n_batches):
            start_idx = batch_size * batch_idx
            end_idx = batch_size * (batch_idx + 1)

            ims = images[start_idx:end_idx]
            ims = to_cuda(ims)
            activations = self.inception_network(ims)
            activations = activations.detach().cpu().numpy()
            assert activations.shape == (ims.shape[0], 2048), "Expexted output shape to be: {}, but was: {}".format(
                (ims.shape[0], 2048), activations.shape)
            inception_activations[start_idx:end_idx, :] = activations
        return inception_activations

    # Modified from: https://github.com/bioinf-jku/TTUR/blob/master/fid.py
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
                
        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1 : Numpy array containing the activations of the pool_3 layer of the
                inception net ( like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
                on an representive data set.
        -- sigma1: The covariance matrix over activations of the pool_3 layer for
                generated samples.
        -- sigma2: The covariance matrix over activations of the pool_3 layer,
                precalcualted on an representive data set.

        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

        diff = mu1 - mu2
        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

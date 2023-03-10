#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import yaml
import os
import numpy as np
from rangenet.tasks.semantic.modules.segmentator import *
from rangenet.tasks.semantic.postproc.KNN import KNN
import random
from scipy import linalg
import pickle
from tqdm import trange, tqdm
from util import _map

class FID():
  def __init__(self, train_dataset, dataset_name, lidar, max_sample=1000, batch_size=8):

    self.ARCH = yaml.safe_load(open('configs/arch_cfg.yaml', 'r'))
    self.DATA = yaml.safe_load(open('configs/data_cfg.yaml', 'r'))
    self.path = './'
    self.sensor_img_means = torch.tensor(self.ARCH["dataset"]["sensor"]["img_means"], dtype=torch.float)
    self.sensor_img_stds = torch.tensor(self.ARCH["dataset"]["sensor"]["img_stds"], dtype=torch.float)
    self.batch_size = batch_size
    ds = train_dataset
    n_samples = min(max_sample, len(train_dataset))
    stat_dir = os.path.join('fid_stats', f'fid_{dataset_name}.pkl')
    # parameters
    self.modeldir = os.path.join('rangenet', 'rangenet_weights')
    self.n_classes = len(self.DATA["learning_map_inv"])
    self.lidar = lidar 
    # concatenate the encoder and the head
    with torch.no_grad():
      self.model = Segmentator(self.ARCH, self.n_classes, self.modeldir)

    # use knn post processing?
    self.post = None
    if self.ARCH["post"]["KNN"]["use"]:
      self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                       self.n_classes)

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model.to(self.device)

    if os.path.isfile(stat_dir):
        stat = pickle.load(open(stat_dir, 'rb'))
        self.mu_train, self.sigma_train = stat['mu'], stat['sigma']
        print('FID stats loaded ...\n')
    else:
        sample_indxs = np.random.choice(range(len(train_dataset)), n_samples, replace=False)
        samples = []
        for ind in tqdm(sample_indxs, desc='gathering real samples for fid'):
            data = ds[ind]
            depth = data['depth'] * (lidar.max_depth - lidar.min_depth) + lidar.min_depth
            points = data['points'] * lidar.max_depth
            vol = torch.cat([depth, points, data['reflectance'], data['mask']], dim=0)
            vol = vol.to(self.device)
            samples.append(vol)
        samples = torch.stack(samples, dim=0)
        self.mu_train , self.sigma_train = self.compute_stats(samples)
        pickle.dump({'mu': self.mu_train, 'sigma':self.sigma_train}, open(stat_dir, 'wb'))
        print('FID stats saved ...\n')

  def compute_stats(self, data_tensor):
    feature_array = self.compute_range_net_features(data_tensor)
    _, C, H, W = feature_array.shape
    random.seed(0)
    indices = range(4096)
    # indices = random.sample(range(0, C * H * W), 4096)
    all_activations = []
    for f in feature_array:
      all_activations.append(f.reshape((-1))[indices])
    all_activations = np.stack(all_activations, axis=0)
    mu = np.mean(all_activations, axis=0)
    sigma = np.cov(all_activations, rowvar=False)
    return mu, sigma

  def compute_range_net_features(self, data_tensor):
    n_batch = np.ceil(len(data_tensor) / self.batch_size)
    features_list = []
    for i in trange(int(n_batch), desc='extracting features for fid'):
      data = data_tensor[i * self.batch_size: (i + 1) * self.batch_size]
      in_vol = data[:, :5]
      proj_mask = data[:, [5]]
      in_vol = (in_vol - self.sensor_img_means[None, :, None, None].to(self.device) ) / self.sensor_img_stds[None, :, None, None].to(self.device)
      in_vol = in_vol * proj_mask
      _, feature = self.model(in_vol, proj_mask)
      # a = out[0].argmax(dim=0).detach().cpu().numpy();import matplotlib.pyplot as plt
      # plt.imshow(_map(_map(a, self.DATA['learning_map_inv']), self.DATA['color_map'])[..., ::-1]);plt.show()
      features_list.append(feature.detach().cpu().numpy())
    
    return np.concatenate(features_list, axis=0)

  
  def fid_score(self, samples):
        # list of tensors in cpu

        assert samples.shape[0] > 1 , 'for FID num of samples must be greater than one'
        # batch_size = min(batch_size, samples.shape[0])
        mu , sigma = self.compute_stats(samples)
        fid = self.calculate_frechet_distance(self.mu_train, self.sigma_train, mu , sigma)
        return fid

        #proj_argmax.tofile(path)
  def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
      """Numpy implementation of the Frechet Distance.
      The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
      and X_2 ~ N(mu_2, C_2) is
              d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
      Stable version by Dougal J. Sutherland.
      Params:
      -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
      -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
      -- sigma1: The covariance matrix over activations for generated samples.
      -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.
      Returns:
      --   : The Frechet Distance.
      """

      mu1 = np.atleast_1d(mu1)
      mu2 = np.atleast_1d(mu2)

      sigma1 = np.atleast_2d(sigma1)
      sigma2 = np.atleast_2d(sigma2)

      assert mu1.shape == mu2.shape, \
          'Training and test mean vectors have different lengths'
      assert sigma1.shape == sigma2.shape, \
          'Training and test covariances have different dimensions'

      diff = mu1 - mu2

      # Product might be almost singular
      covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
      if not np.isfinite(covmean).all():
          msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
          print(msg)
          offset = np.eye(sigma1.shape[0]) * eps
          covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

      # Numerical error might give slight imaginary component
      if np.iscomplexobj(covmean):
          if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
              m = np.max(np.abs(covmean.imag))
              raise ValueError('Imaginary component {}'.format(m))
          covmean = covmean.real

      tr_covmean = np.trace(covmean)

      return (diff.dot(diff) + np.trace(sigma1)
              + np.trace(sigma2) - 2 * tr_covmean)
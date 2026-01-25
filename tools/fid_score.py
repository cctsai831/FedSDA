# """Calculates the Frechet Inception Distance (FID) to evalulate GANs

# The FID metric calculates the distance between two distributions of images.
# Typically, we have summary statistics (mean & covariance matrix) of one
# of these distributions, while the 2nd distribution is given by a GAN.

# When run as a stand-alone program, it compares the distribution of
# images that are stored as PNG/JPEG at a specified location with a
# distribution given by summary statistics (in pickle format).

# The FID is calculated by assuming that X_1 and X_2 are the activations of
# the pool_3 layer of the inception net for generated samples and real world
# samples respectively.

# See --help to see further details.

# Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
# of Tensorflow

# Copyright 2018 Institute of Bioinformatics, JKU Linz

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# """

import numpy as np
from scipy import linalg
import glob

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
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

def mu_sigma(features):
    return np.mean(features, axis=0), np.cov(features, rowvar=False)

def stain_2_m_s(stains_H, stains_E):
    mu_H, sigma_H = mu_sigma(stains_H)
    mu_E, sigma_E = mu_sigma(stains_E)
    return mu_H, sigma_H, mu_E, sigma_E

def average_wasserstein(X, Y):
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    dists = []
    for i in range(X.shape[1]):
        x_sorted, _ = torch.sort(X[:, i])
        y_sorted, _ = torch.sort(Y[:, i])
        dist = torch.mean(torch.abs(x_sorted - y_sorted))
        dists.append(dist)
    return torch.Tensor.numpy(torch.mean(torch.stack(dists)))

import torch
def compute_mmd(X, Y, gamma=1.0):
    """
    X, Y: [N, D] torch.Tensor
    gamma: kernel bandwidth
    """
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    XX = torch.cdist(X, X).pow(2)
    YY = torch.cdist(Y, Y).pow(2)
    XY = torch.cdist(X, Y).pow(2)

    K_XX = torch.exp(-gamma * XX)
    K_YY = torch.exp(-gamma * YY)
    K_XY = torch.exp(-gamma * XY)

    mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return torch.Tensor.numpy(mmd)

from sklearn.metrics.pairwise import rbf_kernel
def compute_mmd_npy(X, Y, gamma=1.0):
    XX = rbf_kernel(X, X, gamma=gamma)
    YY = rbf_kernel(Y, Y, gamma=gamma)
    XY = rbf_kernel(X, Y, gamma=gamma)
    return np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)

import os
def stain_fid(file_path, client):
    
    print("Loading Eval Stains ...")
    # eval stains
    paths = glob.glob(file_path+"/*.npy")
    stains = np.stack([np.load(path) for path in paths])
    
    print("Loading Ori Stains ...")
    # ori stains
    ori_stain_path = f"/workspace/FL/dataset/center_{int(int(client)+1)}_stain/"
    ori_stain_paths = [os.path.join(ori_stain_path, file) for file in os.listdir(ori_stain_path) if file.endswith('.npy')]
    ori_stains = np.stack([np.load(path) for path in ori_stain_paths])
    assert ori_stains.shape==stains.shape

    print("Split Stains to H & E matrics ...")
    # H & E matric
    stains_H, stains_E = stains[:, 0:1, :].squeeze(1), stains[:, 1:2, :].squeeze(1)
    ori_stains_H, ori_stains_E = ori_stains[:, 0:1, :].squeeze(1), ori_stains[:, 1:2, :].squeeze(1)

    # eval stains mu & digma
    mu_H, sigma_H, mu_E, sigma_E = stain_2_m_s(stains_H, stains_E)

    # ori stains  mu & digma
    ori_mu_H=np.load(f"/workspace/FL/U-ViT/tools/mu_sigma/{client}_mu_H.npy")
    ori_mu_E=np.load(f"/workspace/FL/U-ViT/tools/mu_sigma/{client}_mu_E.npy")
    ori_sigma_H=np.load(f"/workspace/FL/U-ViT/tools/mu_sigma/{client}_sigma_H.npy")
    ori_sigma_E=np.load(f"/workspace/FL/U-ViT/tools/mu_sigma/{client}_sigma_E.npy")

    print("FID ...")
    # FID
    fid1 = calculate_frechet_distance(mu_H, sigma_H, ori_mu_H, ori_sigma_H, eps=1e-6)
    fid2 = calculate_frechet_distance(mu_E, sigma_E, ori_mu_E, ori_sigma_E, eps=1e-6)
    print("MMD ...")
    # MMD
    mmd1 = compute_mmd(stains_H, ori_stains_H)
    mmd2 = compute_mmd(stains_E, ori_stains_E)
    print("WD-1D ...")
    # WD-1D
    wd1 = average_wasserstein(stains_H, ori_stains_H)
    wd2 = average_wasserstein(stains_E, ori_stains_E)

    return fid1, fid2, (fid1+fid2)/2, mmd1, mmd2, (mmd1+mmd2)/2, wd1, wd2, (wd1+wd2)/2
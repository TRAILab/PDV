#!/usr/bin/env python

import unittest

import numpy as np
from sklearn.neighbors import KernelDensity
import torch

from pcdet.utils.kde_utils import GaussianKernelDensityEstimation


class TestGaussianKernelDensityEstimation(unittest.TestCase):
    def sample_multimodal(self, n, nsample, d, means, std_devs, weights):
        # Assume equal weighting between distributions
        mixture_ids = np.random.choice(len(means), size=(n, nsample), replace=True, p=weights)
        mixture_means = means[mixture_ids][..., None]
        mixture_std_devs = std_devs[mixture_ids][..., None]
        return np.random.randn(n, nsample, d) * mixture_std_devs + mixture_means

    def test_gaussian_kde_default(self):
        # True distribution params
        weights = np.array([0.3, 0.7])
        means = np.array([0., 5.])
        std_devs = np.array([3., 1.])

        # Input argument samples
        samples = torch.tensor(self.sample_multimodal(1, 16, 3, means, std_devs, weights)).squeeze()
        balls_idx = torch.ones((list(samples.shape[:-1])), dtype=torch.bool)
        est_points = torch.randn(5, 3)

        # Kernel Density Estimation params
        bandwidth = 3.
        kde = GaussianKernelDensityEstimation(bandwidth=bandwidth)
        kde_truth = KernelDensity(bandwidth=bandwidth, kernel='gaussian')

        kde_truth.fit(samples)
        expected_log = torch.tensor(kde_truth.score_samples(est_points))

        estimated = kde.score_samples(samples.unsqueeze(0),
                                      balls_idx.unsqueeze(0),
                                      est_points.unsqueeze(0))

        self.assertTrue(torch.allclose(expected_log, estimated.log()))

    def test_gaussian_kde_multiple_default(self):
        # True distribution params
        weights = np.array([0.3, 0.7])
        means = np.array([0., 5.])
        std_devs = np.array([3., 1.])

        # Input argument samples
        samples = torch.tensor(self.sample_multimodal(2, 16, 3, means, std_devs, weights))
        balls_idx = torch.ones(list(samples.shape[:-1]), dtype=torch.bool)
        est_points = torch.randn(samples.shape[0], 5, 3)

        # Kernel Density Estimation params
        bandwidth = 3.
        kde = GaussianKernelDensityEstimation(bandwidth=bandwidth)
        kde_truth = KernelDensity(bandwidth=bandwidth, kernel='gaussian')

        # Get ground truth
        kde_truth.fit(samples[0])
        expected_log_1 = torch.tensor(kde_truth.score_samples(est_points[0]))
        kde_truth.fit(samples[1])
        expected_log_2 = torch.tensor(kde_truth.score_samples(est_points[1]))
        expected_log = torch.stack((expected_log_1, expected_log_2), axis=0)

        # Pytorch implementation
        # Batch size 1, N = 2
        estimated = kde.score_samples(samples,
                                      balls_idx,
                                      est_points)

        self.assertTrue(torch.allclose(expected_log, estimated.log()))

    def test_gaussian_kde_mask_default(self):
        # True distribution params
        weights = np.array([0.3, 0.7])
        means = np.array([0., 5.])
        std_devs = np.array([3., 1.])

        # Input argument samples
        samples = torch.tensor(self.sample_multimodal(2, 16, 3, means, std_devs, weights))
        balls_idx = torch.ones(list(samples.shape[:-1]), dtype=torch.bool)
        est_points = torch.randn(samples.shape[0], 5, 3)

        # Kernel Density Estimation params
        bandwidth = 3.
        kde = GaussianKernelDensityEstimation(bandwidth=bandwidth)

        # Mask out some samples
        balls_idx[..., -2:] = False

        # Check if truncated and masked inputs give same results
        estimated_truncated = kde.score_samples(samples[..., :-2, :], balls_idx[..., :-2], est_points)
        estimated_masked = kde.score_samples(samples, balls_idx, est_points)
        self.assertTrue(torch.allclose(estimated_truncated.log(), estimated_masked.log()))

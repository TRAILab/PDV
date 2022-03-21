#!/usr/bin/env python

import unittest
from functools import partial

import torch
import numpy as np
from spconv import SparseConvTensor

from pcdet.utils.voxel_aggregation_utils import (get_centroids_per_voxel_layer, get_overlapping_voxel_indices,
                                                 get_centroid_per_voxel,
                                                 get_voxel_indices_to_voxel_list_index,
                                                 get_nonempty_voxel_feature_indices)


class TestOverlappingVoxelIndices(unittest.TestCase):
    def test_get_overlapping_voxel_indices_default(self):
        downsample_times = 1
        voxel_size = [0.1, 0.1, 0.1]
        point_cloud_range = [0., 0., 0., 1., 1., 1.]

        get_overlapping_voxel_indices_default = partial(
            get_overlapping_voxel_indices,
            downsample_times=downsample_times,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range)

        point_coords_1 = torch.tensor([[0., 0., 0.]])
        voxel_indices_1 = get_overlapping_voxel_indices_default(point_coords_1).numpy()
        voxel_indices_1_exp = np.array([[0, 0, 0]])
        np.testing.assert_equal(voxel_indices_1, voxel_indices_1_exp)

        point_coords_2 = torch.tensor([[0.13, 0.25, 0.08]])
        voxel_indices_2 = get_overlapping_voxel_indices_default(point_coords_2).numpy()
        voxel_indices_2_exp = np.array([[1, 2, 0]])
        np.testing.assert_equal(voxel_indices_2, voxel_indices_2_exp)

        point_coords_3 = torch.tensor([[1.2, 0.25, 0.08]])
        voxel_indices_3 = get_overlapping_voxel_indices_default(point_coords_3).numpy()
        voxel_indices_3_exp = np.array([[-1, -1, -1]])
        np.testing.assert_equal(voxel_indices_3, voxel_indices_3_exp)

        point_coords_4 = torch.tensor([[0.13, -0.5, 0.08]])
        voxel_indices_4 = get_overlapping_voxel_indices_default(point_coords_4).numpy()
        voxel_indices_4_exp = np.array([[-1, -1, -1]])
        np.testing.assert_equal(voxel_indices_4, voxel_indices_4_exp)

        point_coords_5 = torch.tensor([[0.13 , -0.5 , 0.08],
                                       [-0.01,  0.  , 0.  ],
                                       [0.53 ,  0.21, 0.93]])
        voxel_indices_5 = get_overlapping_voxel_indices_default(point_coords_5).numpy()
        voxel_indices_5_exp = np.array([[-1, -1, -1],
                                        [-1, -1, -1],
                                        [ 5,  2,  9]])
        np.testing.assert_equal(voxel_indices_5, voxel_indices_5_exp)

    def test_get_overlapping_voxel_indices_downsample_times(self):
        downsample_times = 2
        voxel_size = [0.1, 0.1, 0.1]
        point_cloud_range = [0., 0., 0., 1., 1., 1.]

        get_overlapping_voxel_indices_downsample_times = partial(
            get_overlapping_voxel_indices,
            downsample_times=downsample_times,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range)

        point_coords_1 = torch.tensor([[0., 0., 0.]])
        voxel_indices_1 = get_overlapping_voxel_indices_downsample_times(point_coords_1).numpy()
        voxel_indices_1_exp = np.array([[0, 0, 0]])
        np.testing.assert_equal(voxel_indices_1, voxel_indices_1_exp)

        point_coords_2 = torch.tensor([[0.13, 0.9, 0.79]])
        voxel_indices_2 = get_overlapping_voxel_indices_downsample_times(point_coords_2).numpy()
        voxel_indices_2_exp = np.array([[0, 4, 3]])
        np.testing.assert_equal(voxel_indices_2, voxel_indices_2_exp)

        point_coords_3 = torch.tensor([[1.19, 0.25, 0.08]])
        voxel_indices_3 = get_overlapping_voxel_indices_downsample_times(point_coords_3).numpy()
        voxel_indices_3_exp = np.array([[-1, -1, -1]])
        np.testing.assert_equal(voxel_indices_3, voxel_indices_3_exp)

    def test_get_overlapping_voxel_indices_voxel_size(self):
        downsample_times = 1
        voxel_size = [0.1, 0.5, 0.3]
        point_cloud_range = [0., 0., 0., 1., 1., 1.]

        get_overlapping_voxel_indices_voxel_size = partial(
            get_overlapping_voxel_indices,
            downsample_times=downsample_times,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range)

        point_coords_1 = torch.tensor([[0., 0., 0.]])
        voxel_indices_1 = get_overlapping_voxel_indices_voxel_size(point_coords_1).numpy()
        voxel_indices_1_exp = np.array([[0, 0, 0]])
        np.testing.assert_equal(voxel_indices_1, voxel_indices_1_exp)

        point_coords_2 = torch.tensor([[0.13, 0.9, 0.79]])
        voxel_indices_2 = get_overlapping_voxel_indices_voxel_size(point_coords_2).numpy()
        voxel_indices_2_exp = np.array([[1, 1, 2]])
        np.testing.assert_equal(voxel_indices_2, voxel_indices_2_exp)

        point_coords_3 = torch.tensor([[1.2, 0.25, 0.08]])
        voxel_indices_3 = get_overlapping_voxel_indices_voxel_size(point_coords_3).numpy()
        voxel_indices_3_exp = np.array([[-1, -1, -1]])
        np.testing.assert_equal(voxel_indices_3, voxel_indices_3_exp)

    def test_get_overlapping_voxel_indices_pc_range(self):
        downsample_times = 1
        voxel_size = [0.1, 0.1, 0.1]
        point_cloud_range = [-2., 0., -1., 1., 1., 1.]

        get_overlapping_voxel_indices_default = partial(
            get_overlapping_voxel_indices,
            downsample_times=downsample_times,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range)

        point_coords_1 = torch.tensor([[0., 0., 0.]])
        voxel_indices_1 = get_overlapping_voxel_indices_default(point_coords_1).numpy()
        voxel_indices_1_exp = np.array([[20, 0, 10]])
        np.testing.assert_equal(voxel_indices_1, voxel_indices_1_exp)

        point_coords_2 = torch.tensor([[0.13, 0.25, 0.08]])
        voxel_indices_2 = get_overlapping_voxel_indices_default(point_coords_2).numpy()
        voxel_indices_2_exp = np.array([[21, 2, 10]])
        np.testing.assert_equal(voxel_indices_2, voxel_indices_2_exp)

        point_coords_3 = torch.tensor([[1.2, 0.25, 0.08]])
        voxel_indices_3 = get_overlapping_voxel_indices_default(point_coords_3).numpy()
        voxel_indices_3_exp = np.array([[-1, -1, -1]])
        np.testing.assert_equal(voxel_indices_3, voxel_indices_3_exp)

        point_coords_4 = torch.tensor([[0.13, -0.5, 0.08]])
        voxel_indices_4 = get_overlapping_voxel_indices_default(point_coords_4).numpy()
        voxel_indices_4_exp = np.array([[-1, -1, -1]])
        np.testing.assert_equal(voxel_indices_4, voxel_indices_4_exp)

        point_coords_5 = torch.tensor([[0.13, -0.5 , 0.08],
                                       [-2.1,  0.  , 0.  ],
                                       [0.53,  0.21, 0.93]])
        voxel_indices_5 = get_overlapping_voxel_indices_default(point_coords_5).numpy()
        voxel_indices_5_exp = np.array([[ -1,  -1,  -1],
                                        [ -1,  -1,  -1],
                                        [ 25,  2,  19]])
        np.testing.assert_equal(voxel_indices_5, voxel_indices_5_exp)


class TestGetVoxelIndicesToVoxelListIndex(unittest.TestCase):
    def test_get_voxel_indices_to_voxel_list_index_default(self):
        dense_voxel_grid = torch.zeros(2, 3, 3, 3, 1)
        dense_voxel_grid[0, 0, 1, 2, 0] = 5.
        dense_voxel_grid[0, 1, 1, 1, 0] = 3.
        dense_voxel_grid[1, 2, 0, 0, 0] = 1.
        sparse_tensor = SparseConvTensor.from_dense(dense_voxel_grid)

        true_voxel_hash_map = torch.zeros(2, 3, 3, 3, dtype=torch.long)
        true_voxel_hash_map[0, 0, 1, 2] = 1
        true_voxel_hash_map[0, 1, 1, 1] = 2
        true_voxel_hash_map[1, 2, 0, 0] = 3
        predicted_voxel_hash_map = get_voxel_indices_to_voxel_list_index(sparse_tensor)
        self.assertTrue(torch.all(torch.eq(true_voxel_hash_map, predicted_voxel_hash_map)))


class TestGetNonemptyVoxelFeatureIndices(unittest.TestCase):
    def test_get_nonempty_voxel_feature_indices_default(self):
        dense_voxel_grid = torch.zeros(2, 3, 3, 3, 1)
        dense_voxel_grid[0, 0, 1, 2, 0] = 5.
        dense_voxel_grid[0, 1, 1, 1, 0] = 3.
        dense_voxel_grid[1, 2, 0, 0, 0] = 1.
        # Permute from (bxyzc) to (bzxyc)
        dense_voxel_grid = dense_voxel_grid.permute(0, 3, 2, 1, 4)
        sparse_tensor = SparseConvTensor.from_dense(dense_voxel_grid)

        # Voxel grid parameters
        downsample_factor = 1
        voxel_size = [0.1, 0.1, 0.1]
        point_cloud_range = [0., 0., 0., 0.3, 0.3, 0.3]
        points = torch.Tensor([
            [0, 0., 0., 0.],
            [0, 0., 0.1, 0.2],
            [1, 0., 0., 0.],
            [1, 0., 0.1, 0.2],
            [1, 0.2, 0., 0.],
        ])
        _, voxel_indices_all = get_centroids_per_voxel_layer(points, ['xconv'], {'xconv': downsample_factor}, voxel_size, point_cloud_range)
        voxel_indices = voxel_indices_all['xconv']
        true_indices = torch.FloatTensor([1, 2])
        true_mask = torch.BoolTensor([False, True, False, True, False])
        predicted_indices, predicted_mask = get_nonempty_voxel_feature_indices(voxel_indices, sparse_tensor)
        self.assertTrue(torch.all(torch.eq(true_indices, predicted_indices)))
        self.assertTrue(torch.all(torch.eq(true_mask, predicted_mask)))


class TestGetCentroidPerVoxel(unittest.TestCase):
    def test_get_centroid_per_voxel_default(self):
        points = torch.Tensor([
            [0., 0.1, 0.4, 0.1],
            [0., 0.5, 0.4, 0.3],
            [1., 0.2, 0.2, -0.1],
            [1., 0.0, 0.0, 0.0],
            [1., 0.9, 1.4, 0.3],
            [1., 0.3, 1.5, 0.2]
        ])
        voxel_feature_indices = torch.LongTensor([
            [0, 5, 3, 1],
            [0, 15, 2, 1],
            [1, 5, 3, 1],
            [1, 9, 2, 4],
            [1, 3, 6, 9],
            [1, 3, 6, 9]
        ])

        true_centroids = torch.Tensor([
            [0., 0.1, 0.4, 0.1],
            [0., 0.5, 0.4, 0.3],
            [1., 0.6, 1.45, 0.25],
            [1., 0.2, 0.2, -0.1],
            [1., 0., 0., 0.]
        ])
        true_unique_indices = torch.LongTensor([
            [0, 5, 3, 1],
            [0, 15, 2, 1],
            [1, 3, 6, 9],
            [1, 5, 3, 1],
            [1, 9, 2, 4],
        ])
        true_count = torch.LongTensor([1, 1, 2, 1, 1])

        predicted_centroids, predicted_unique_indices, predicted_count = get_centroid_per_voxel(points, voxel_feature_indices)
        self.assertTrue(torch.all(torch.isclose(true_centroids, predicted_centroids)))
        self.assertTrue(torch.all(torch.eq(true_unique_indices, predicted_unique_indices)))
        self.assertTrue(torch.all(torch.eq(true_count, predicted_count)))

    def test_centroid_location(self):
        # Voxel grid parameters
        downsample_times = 1
        voxel_size = [0.1, 0.1, 0.1]
        point_cloud_range = [0., 0., 0., 1., 1., 1.]

        # Points
        points = torch.Tensor([
            [0., 0.1, 0.4, 0.1],
            [0., 0.5, 0.4, 0.3],
            [1., 0.2, 0.2, 0.1],
            [1., 0.0, 0.0, 0.0],
            [1., 0.9, 0.5, 0.3],
            [1., 0.96, 0.5, 0.38]
        ])
        voxel_indices = torch.LongTensor([
            [1, 4, 1],
            [5, 4, 3],
            [2, 2, 1],
            [0, 0, 0],
            [9, 5, 3],
            [9, 5, 3]
        ])

        true_centroids = torch.Tensor([
            [1., 0.0, 0.0, 0.0],
            [0., 0.1, 0.4, 0.1],
            [1., 0.2, 0.2, 0.1],
            [0., 0.5, 0.4, 0.3],
            [1., 0.93, 0.5, 0.34],
        ])
        true_unique_indices = torch.LongTensor([
            [0, 0, 0],
            [1, 4, 1],
            [2, 2, 1],
            [5, 4, 3],
            [9, 5, 3],
        ])
        predicted_centroids, predicted_unique_indices, predicted_count = get_centroid_per_voxel(points, voxel_indices)
        self.assertTrue(torch.all(torch.isclose(true_centroids, predicted_centroids)))
        self.assertTrue(torch.all(torch.eq(true_unique_indices, predicted_unique_indices)))

        true_centroid_voxel_indices = torch.LongTensor([
            [0, 0, 0],
            [1, 4, 1],
            [2, 2, 1],
            [5, 4, 3],
            [9, 5, 3],
        ])
        predicted_centroid_voxel_indices = get_overlapping_voxel_indices(
            predicted_centroids[:, 1:4],
            downsample_times=downsample_times,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range
        )
        self.assertTrue(torch.all(torch.eq(true_centroid_voxel_indices, predicted_centroid_voxel_indices)))


class TestGetCentroidPerVoxelLayer(unittest.TestCase):
    def test_get_centroids_per_voxel_layer(self):
        # Voxel grid parameters
        voxel_size = [0.1, 0.1, 0.1]
        point_cloud_range = [0., 0., 0., 1., 1., 1.]
        feature_locations = ['x_conv1', 'x_conv2']
        multi_scale_3d_strides = {'x_conv1': 1, 'x_conv2': 2}

        # Points
        points = torch.Tensor([
            [0., 0., 0.4, 0.1],
            [0., 0.1, 0.4, 0.1],
            [0., 0.15, 0.43, 0.12],
            [0., 0.1, 0.5, 0.1],
            [0., 0.5, 0.4, 0.3],
            [1., 0.0, 0.0, 0.0],
            [1., 0.2, 0.2, 0.1],
            [1., 0.15, 0.43, 0.12],
            [1., 0.9, 0.5, 0.3],
            [1., 0.96, 0.5, 0.38]
        ])

        true_centroids_all = {
            'x_conv1': torch.Tensor([
                [0., 0., 0.4, 0.1],
                [0., 0.125, 0.415, 0.11],
                [0., 0.1, 0.5, 0.1],
                [0., 0.5, 0.4, 0.3],
                [1., 0.0, 0.0, 0.0],
                [1., 0.2, 0.2, 0.1],
                [1., 0.15, 0.43, 0.12],
                [1., 0.93, 0.5, 0.34],
            ]),
            'x_conv2': torch.Tensor([
                [0., 0.0875, 0.4325, 0.105],
                [0., 0.5, 0.4, 0.3],
                [1., 0., 0., 0.],
                [1., 0.2, 0.2, 0.1],
                [1., 0.15, 0.43, 0.12],
                [1., 0.93, 0.5, 0.34]
            ])
        }
        true_centroid_voxel_idxs_all = {
            'x_conv1': torch.LongTensor([
                [0, 1, 4, 0],
                [0, 1, 4, 1],
                [0, 1, 5, 1],
                [0, 3, 4, 5],
                [1, 0, 0, 0],
                [1, 1, 2, 2],
                [1, 1, 4, 1],
                [1, 3, 5, 9]
            ]),
            'x_conv2': torch.LongTensor([
                [0, 0, 2, 0],
                [0, 1, 2, 2],
                [1, 0, 0, 0],
                [1, 0, 1, 1],
                [1, 0, 2, 0],
                [1, 1, 2, 4]
            ])
        }
        centroids_all, centroid_voxel_idxs_all = get_centroids_per_voxel_layer(points, feature_locations, multi_scale_3d_strides, voxel_size, point_cloud_range)
        for feature_location in feature_locations:
            self.assertTrue(torch.all(torch.isclose(true_centroids_all[feature_location],
                                                    centroids_all[feature_location])))
            self.assertTrue(torch.all(torch.isclose(true_centroid_voxel_idxs_all[feature_location],
                                                    centroid_voxel_idxs_all[feature_location])))

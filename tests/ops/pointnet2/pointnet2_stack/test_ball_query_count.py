#!/usr/bin/env python

import unittest
import time

import torch

from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils


class TestBallQueryCount(unittest.TestCase):
    def test_ball_query_count_default(self):
        num_points = 2000
        num_features = 8
        num_new_points = 200

        points = torch.randn(num_points, 3)
        features = torch.randn(num_points, num_features)
        batch_number = torch.zeros(num_points, 1)
        batch_number[(num_points // 2):] = 1
        xyz = torch.cat((batch_number, points), dim=1)
        _, xyz_batch_cnt = batch_number.unique(return_counts=True)


        new_points = torch.randn(num_new_points, 3)
        new_batch_number = torch.zeros(num_new_points, 1)
        new_batch_number[(num_new_points // 2):] = 1
        new_xyz = torch.cat((new_batch_number, new_points), dim=1)
        _, new_xyz_batch_cnt = new_batch_number.unique(return_counts=True)

        query_and_group = pointnet2_utils.QueryAndGroup(2, 16, use_xyz=True, use_density=False)
        query_and_group_density = pointnet2_utils.QueryAndGroup(2, 16, use_xyz=True, use_density=True)

        xyz = xyz.cuda()
        xyz_batch_cnt = xyz_batch_cnt.cuda().int()
        new_xyz = new_xyz.cuda()
        new_xyz_batch_cnt = new_xyz_batch_cnt.cuda().int()
        features = features.cuda()

        start = time.time()
        new_features, ball_idxs = query_and_group(xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features)
        end = time.time()
        query_and_group_time = end - start
        start_density = time.time()
        new_features_density, ball_idxs_density = query_and_group_density(xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features)
        end_density = time.time()
        query_and_group_density_time = end_density - start_density

        print(f'Query and group time: {query_and_group_time} (s), {query_and_group_density_time} (s)')

        self.assertTrue(torch.allclose(new_features[:, :3], new_features_density[:, :3]))
        self.assertTrue(torch.allclose(new_features[:, 4:], new_features_density[:, 5:]))
        self.assertTrue(torch.allclose(ball_idxs, ball_idxs_density ))

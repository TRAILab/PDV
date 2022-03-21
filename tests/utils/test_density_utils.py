#!/usr/bin/env python

import unittest
import math

import torch

from pcdet.utils.density_utils import find_num_points_per_part, find_num_points_per_part_multi


class TestFindNumPointsPerPart(unittest.TestCase):
    def test_find_num_points_per_part_default(self):
        points = torch.tensor([
            [0, 0, 0.1],
            [1.4, 0.8, 0],
            [3, 2, 4],
            [0, 0, 2]
        ], dtype=torch.float32).cuda()
        batch_idx = torch.zeros(points.shape[0], 1).cuda()
        boxes = torch.tensor([
            [0, 0, 0, 2, 2, 2, 0],
            [10, 1, 2, 3, 3, 3, 0]
        ], dtype=torch.float32).cuda()

        batch_points = torch.cat((batch_idx, points), dim=1)
        batch_boxes = boxes.unsqueeze(0)
        grid_size = 1

        points_per_part = find_num_points_per_part(batch_points, batch_boxes, grid_size).squeeze(-1).squeeze(-1).cpu()
        true_points_per_part = torch.tensor([[[1], [0]]], dtype=torch.int64)
        self.assertTrue(torch.all(torch.eq(true_points_per_part, points_per_part)))

    def test_find_num_points_per_part_overlap(self):
        points = torch.tensor([
            [0, 0, 0.1],
            [1.4, 0.8, 0],
            [3, 2, 4],
            [0, 0, 0]
        ], dtype=torch.float32).cuda()
        batch_idx = torch.zeros(points.shape[0], 1).cuda()
        boxes = torch.tensor([
            [0, 0, 0, 2, 2, 2, 0],
            [0, 0, 0, 1, 1, 1, 0]
        ], dtype=torch.float32).cuda()

        batch_points = torch.cat((batch_idx, points), dim=1)
        batch_boxes = boxes.unsqueeze(0)
        grid_size = 1

        points_per_part = find_num_points_per_part(batch_points, batch_boxes, grid_size).squeeze(-1).squeeze(-1).cpu()
        true_points_per_part = torch.tensor([[[2], [0]]], dtype=torch.int64)
        self.assertTrue(torch.all(torch.eq(true_points_per_part, points_per_part)))

    def test_find_num_points_per_part_angles(self):
        points = torch.tensor([
            [0, 0, 0],
            [1.4, 0.8, 0],
        ], dtype=torch.float32).cuda()
        batch_idx = torch.zeros(points.shape[0], 1).cuda()
        boxes = torch.tensor([
            [0, 0, 0, 2, 2, 2, math.pi / 4],
        ], dtype=torch.float32).cuda()

        batch_points = torch.cat((batch_idx, points), dim=1)
        batch_boxes = boxes.unsqueeze(0)
        grid_size = 1

        points_per_part = find_num_points_per_part(batch_points, batch_boxes, grid_size).squeeze(-1).squeeze(-1).cpu()
        true_points_per_part = torch.tensor([[[1]]])
        self.assertTrue(torch.all(torch.eq(true_points_per_part, points_per_part)))

    def test_find_num_points_per_part_multi_part(self):
        points = torch.tensor([
            [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5],
        ], dtype=torch.float32).cuda()
        batch_idx = torch.zeros(points.shape[0], 1).cuda()
        boxes = torch.tensor([
            [0, 0, 0, 2, 2, 2, 0],
        ], dtype=torch.float32).cuda()

        batch_points = torch.cat((batch_idx, points), dim=1)
        batch_boxes = boxes.unsqueeze(0)
        grid_size = 2

        points_per_part = find_num_points_per_part(batch_points, batch_boxes, grid_size).squeeze().cpu()
        true_points_per_part = torch.ones((grid_size, grid_size, grid_size), dtype=torch.int64)
        true_points_per_part[0, 0, 0] = 0
        true_points_per_part[0, 1, 0] = 0
        self.assertTrue(torch.all(torch.eq(true_points_per_part, points_per_part)))

    def test_find_num_points_per_part_multi_part_angles(self):
        points = torch.tensor([
            [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5],
        ], dtype=torch.float32).cuda()
        batch_idx = torch.zeros(points.shape[0], 1).cuda()
        boxes = torch.tensor([
            [0, 0, 0, 2, 2, 2, math.pi/2],
        ], dtype=torch.float32).cuda()

        batch_points = torch.cat((batch_idx, points), dim=1)
        batch_boxes = boxes.unsqueeze(0)
        grid_size = 2

        points_per_part = find_num_points_per_part(batch_points, batch_boxes, grid_size).squeeze().cpu()
        true_points_per_part = torch.ones((grid_size, grid_size, grid_size), dtype=torch.int64)
        true_points_per_part[0, 1, 0] = 0
        true_points_per_part[1, 1, 0] = 0
        self.assertTrue(torch.all(torch.eq(true_points_per_part, points_per_part)))

class TestFindNumPointsPerPartMulti(unittest.TestCase):
    def test_find_num_points_per_part_multi_default(self):
        points = torch.tensor([
            [0, 0, 0.1],
            [1.4, 0.8, 0],
            [3, 2, 4],
            [0, 0, 2]
        ], dtype=torch.float32).cuda()
        batch_idx = torch.zeros(points.shape[0], 1).cuda()
        boxes = torch.tensor([
            [0, 0, 0, 2, 2, 2, 0],
            [10, 1, 2, 3, 3, 3, 0]
        ], dtype=torch.float32).cuda()

        batch_points = torch.cat((batch_idx, points), dim=1)
        batch_boxes = boxes.unsqueeze(0)
        grid_size = 1
        max_num_boxes = 3

        points_per_part = find_num_points_per_part_multi(batch_points, batch_boxes, grid_size, max_num_boxes).squeeze(-1).squeeze(-1).cpu()
        true_points_per_part = torch.tensor([[[1], [0]]], dtype=torch.int64)
        self.assertTrue(torch.all(torch.eq(true_points_per_part, points_per_part)))

    def test_find_num_points_per_part_multi_overlap(self):
        points = torch.tensor([
            [0, 0, 0.1],
            [1.4, 0.8, 0],
            [3, 2, 4],
            [0, 0, 0]
        ], dtype=torch.float32).cuda()
        batch_idx = torch.zeros(points.shape[0], 1).cuda()
        boxes = torch.tensor([
            [0, 0, 0, 2, 2, 2, 0],
            [0, 0, 0, 1, 1, 1, 0]
        ], dtype=torch.float32).cuda()

        batch_points = torch.cat((batch_idx, points), dim=1)
        batch_boxes = boxes.unsqueeze(0)
        grid_size = 1
        max_num_boxes = 3

        points_per_part = find_num_points_per_part_multi(batch_points, batch_boxes, grid_size, max_num_boxes).squeeze(-1).squeeze(-1).cpu()
        true_points_per_part = torch.tensor([[[2], [2]]], dtype=torch.int64)
        self.assertTrue(torch.all(torch.eq(true_points_per_part, points_per_part)))

    def test_find_num_points_per_part_angles(self):
        points = torch.tensor([
            [0, 0, 0],
            [1.4, 0.8, 0],
        ], dtype=torch.float32).cuda()
        batch_idx = torch.zeros(points.shape[0], 1).cuda()
        boxes = torch.tensor([
            [0, 0, 0, 2, 2, 2, math.pi / 4],
        ], dtype=torch.float32).cuda()

        batch_points = torch.cat((batch_idx, points), dim=1)
        batch_boxes = boxes.unsqueeze(0)
        grid_size = 1
        max_num_boxes = 3

        points_per_part = find_num_points_per_part_multi(batch_points, batch_boxes, grid_size, max_num_boxes).squeeze(-1).squeeze(-1).cpu()
        true_points_per_part = torch.tensor([[[1]]])
        self.assertTrue(torch.all(torch.eq(true_points_per_part, points_per_part)))

    def test_find_num_points_per_part_multi_part(self):
        points = torch.tensor([
            [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5],
        ], dtype=torch.float32).cuda()
        batch_idx = torch.zeros(points.shape[0], 1).cuda()
        boxes = torch.tensor([
            [0, 0, 0, 2, 2, 2, 0],
        ], dtype=torch.float32).cuda()

        batch_points = torch.cat((batch_idx, points), dim=1)
        batch_boxes = boxes.unsqueeze(0)
        grid_size = 2
        max_num_boxes = 3

        points_per_part = find_num_points_per_part_multi(batch_points, batch_boxes, grid_size, max_num_boxes).squeeze().cpu()
        true_points_per_part = torch.ones((grid_size, grid_size, grid_size), dtype=torch.int64)
        true_points_per_part[0, 0, 0] = 0
        true_points_per_part[0, 1, 0] = 0
        self.assertTrue(torch.all(torch.eq(true_points_per_part, points_per_part)))

    def test_find_num_points_per_part_multi_part_angles(self):
        points = torch.tensor([
            [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5],
        ], dtype=torch.float32).cuda()
        batch_idx = torch.zeros(points.shape[0], 1).cuda()
        boxes = torch.tensor([
            [0, 0, 0, 2, 2, 2, math.pi/2],
        ], dtype=torch.float32).cuda()

        batch_points = torch.cat((batch_idx, points), dim=1)
        batch_boxes = boxes.unsqueeze(0)
        grid_size = 2
        max_num_boxes = 3

        points_per_part = find_num_points_per_part_multi(batch_points, batch_boxes, grid_size, max_num_boxes).squeeze().cpu()
        true_points_per_part = torch.ones((grid_size, grid_size, grid_size), dtype=torch.int64)
        true_points_per_part[0, 1, 0] = 0
        true_points_per_part[1, 1, 0] = 0
        self.assertTrue(torch.all(torch.eq(true_points_per_part, points_per_part)))

    def test_find_num_points_per_multi_part_centroids_default(self):
        points = torch.tensor([
            [0, 0, 0.1],
            [1.4, 0.8, 0],
            [3, 2, 4],
            [0, 0, 2]
        ], dtype=torch.float32).cuda()
        batch_idx = torch.zeros(points.shape[0], 1).cuda()
        boxes = torch.tensor([
            [0, 0, 0, 2, 2, 2, 0],
            [10, 1, 2, 3, 3, 3, 0]
        ], dtype=torch.float32).cuda()

        batch_points = torch.cat((batch_idx, points), dim=1)
        batch_boxes = boxes.unsqueeze(0)
        grid_size = 1
        max_num_boxes = 3

        points_per_part = find_num_points_per_part_multi(batch_points, batch_boxes, grid_size, max_num_boxes, return_centroid=True).squeeze(-2).squeeze(-2).squeeze(-2).cpu()
        # We expect parts with no points to have a centroid estimated at [0, 0, 0]
        true_points_per_part = torch.tensor([[[1, 0, 0, 0.1],
                                              [0, 0, 0, 0]]], dtype=torch.float)
        self.assertTrue(torch.all(torch.eq(true_points_per_part, points_per_part)))

    def test_find_num_points_per_part_multi_centroids_overlap(self):
        points = torch.tensor([
            [0, -0.1, 0.1],
            [1.4, 0.8, 0],
            [3, 2, 4],
            [0, 0.1, 0]
        ], dtype=torch.float32).cuda()
        batch_idx = torch.zeros(points.shape[0], 1).cuda()
        boxes = torch.tensor([
            [0.7, -0.4, 0.3, 2, 2, 2, 0],
            [0, 0, 0, 1, 1, 1, 0]
        ], dtype=torch.float32).cuda()

        batch_points = torch.cat((batch_idx, points), dim=1)
        batch_boxes = boxes.unsqueeze(0)
        grid_size = 1
        max_num_boxes = 3

        points_per_part = find_num_points_per_part_multi(batch_points, batch_boxes, grid_size, max_num_boxes, return_centroid=True).squeeze(-2).squeeze(-2).squeeze(-2).cpu()
        true_points_per_part = torch.tensor([[[2, -0.7, 0.4, -0.25],
                                              [2, 0, 0, 0.05]]], dtype=torch.float)
        self.assertTrue(torch.allclose(true_points_per_part, points_per_part))

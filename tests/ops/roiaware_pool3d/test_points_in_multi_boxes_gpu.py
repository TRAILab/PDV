#!/usr/bin/env python

import unittest

import numpy as np
import torch

from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu, points_in_multi_boxes_gpu


class TestPointsInMultiBoxesGpu(unittest.TestCase):
    def test_points_in_multi_boxes_gpu_default(self):
        points = torch.tensor([
            [0, 0, 0.1],
            [1.4, 0.8, 0],
            [3, 2, 4],
            [0, 0, 2],
        ], dtype=torch.float32).cuda()
        boxes = torch.tensor([
            [0, 0, 0, 2, 2, 2, 0],
        ], dtype=torch.float32).cuda()

        batch_boxes = boxes.unsqueeze(0)
        max_num_boxes = 3

        points_in_multi_boxes = points_in_multi_boxes_gpu(points.unsqueeze(0), batch_boxes, max_num_boxes)
        points_in_boxes = points_in_boxes_gpu(points.unsqueeze(0), batch_boxes)
        true_points_in_multi_boxes = torch.zeros((batch_boxes.shape[0], points.shape[0], max_num_boxes), dtype=torch.int64, device=points.device) - 1
        true_points_in_multi_boxes[0, 0, 0] = 0
        self.assertTrue(torch.all(torch.eq(true_points_in_multi_boxes, points_in_multi_boxes)))
        self.assertTrue(torch.all(torch.eq(points_in_multi_boxes[:, :, 0], points_in_boxes)))

    def test_points_in_multi_boxes_gpu_multi(self):
        points = torch.tensor([
            [0, 0, 0.1],
            [1.4, 0.8, 0],
            [3, 2, 4],
            [0, 0, 2],
            [0, 0, 0]
        ], dtype=torch.float32).cuda()
        boxes = torch.tensor([
            [0, 0, 0, 2, 2, 2, 0],
            [0, 0, 0, 1, 1, 1, 0],
        ], dtype=torch.float32).cuda()

        batch_boxes = boxes.unsqueeze(0)
        max_num_boxes = 3

        points_in_multi_boxes = points_in_multi_boxes_gpu(points.unsqueeze(0), batch_boxes, max_num_boxes)
        true_points_in_multi_boxes = torch.zeros((batch_boxes.shape[0], points.shape[0], max_num_boxes), dtype=torch.int64, device=points.device) - 1
        true_points_in_multi_boxes[0, 0, 0] = 0
        true_points_in_multi_boxes[0, 0, 1] = 1
        true_points_in_multi_boxes[0, -1, 0] = 0
        true_points_in_multi_boxes[0, -1, 1] = 1
        self.assertTrue(torch.all(torch.eq(true_points_in_multi_boxes, points_in_multi_boxes)))

    def test_points_in_multi_boxes_gpu_max(self):
        points = torch.tensor([
            [0, 0, 0.1],
            [1.4, 0.8, 0],
            [3, 2, 4],
            [0, 0, 2],
            [0, 0, 0]
        ], dtype=torch.float32).cuda()
        boxes = torch.tensor([
            [0, 0, 0, 2, 2, 2, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0],
        ], dtype=torch.float32).cuda()

        batch_boxes = boxes.unsqueeze(0)
        max_num_boxes = 3

        points_in_multi_boxes = points_in_multi_boxes_gpu(points.unsqueeze(0), batch_boxes, max_num_boxes)
        true_points_in_multi_boxes = torch.zeros((batch_boxes.shape[0], points.shape[0], max_num_boxes), dtype=torch.int64, device=points.device) - 1
        true_points_in_multi_boxes[0, 0, 0] = 0
        true_points_in_multi_boxes[0, 0, 1] = 1
        true_points_in_multi_boxes[0, 0, 2] = 2
        true_points_in_multi_boxes[0, -1, 0] = 0
        true_points_in_multi_boxes[0, -1, 1] = 1
        true_points_in_multi_boxes[0, -1, 2] = 2

    def test_points_in_multi_boxes_gpu_batch(self):
        points = torch.tensor([
            [
                [6, 2, 4],
                [3, 2, 4],
                [0, 0, 0]
            ],
            [
                [0, 0, 0.1],
                [1.4, 0.8, 0],
                [0, 0, 0]
            ],
        ], dtype=torch.float32).cuda()
        batch_boxes = torch.tensor([
            [
                [5, 5, 5, 2, 2, 2, 0],
                [0, 0, -3, 1, 1, 1, 0],
            ],
            [
                [0, 0, 0, 2, 2, 2, 0],
                [0, 0, 0, 1, 1, 1, 0],
            ],
        ], dtype=torch.float32).cuda()

        max_num_boxes = 3

        points_in_multi_boxes = points_in_multi_boxes_gpu(points, batch_boxes, max_num_boxes)
        true_points_in_multi_boxes = torch.tensor([
            [
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1]
            ],
            [
                [ 0,  1, -1],
                [-1, -1, -1],
                [ 0,  1, -1]
            ]], device=points.device, dtype=torch.int32)
        self.assertTrue(torch.all(torch.eq(true_points_in_multi_boxes, points_in_multi_boxes)))

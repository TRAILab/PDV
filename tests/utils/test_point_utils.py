#!/usr/bin/env python

import unittest

import torch

from pcdet.utils.point_utils import get_padded_points


class TestGetPaddedPoints(unittest.TestCase):
    def test_get_padded_points_default(self):
        points = torch.randn(10, 3)
        point_features = torch.randn(10, 8)
        batch_number = torch.zeros(10, 1)
        batch_number[3:] = 1
        points = torch.cat((batch_number, points), dim=1)

        padded_points, padded_features, valid_points_mask = get_padded_points(points, point_features)
        self.assertEqual(padded_points.shape[0], 2)
        self.assertEqual(padded_features.shape[0], 2)
        self.assertEqual(padded_points.shape[1], 7)
        self.assertEqual(padded_features.shape[1], 7)
        self.assertEqual(padded_points.shape[-1], points.shape[-1] - 1)
        self.assertEqual(padded_features.shape[-1], point_features.shape[-1])
        self.assertTrue(torch.all(torch.isclose(points[:, 1:4], padded_points[valid_points_mask])))
        self.assertTrue(torch.all(torch.isclose(point_features, padded_features[valid_points_mask])))

    def test_get_padded_points_no_features(self):
        points = torch.randn(10, 3)
        batch_number = torch.zeros(10, 1)
        batch_number[3:] = 1
        points = torch.cat((batch_number, points), dim=1)

        padded_points, valid_points_mask = get_padded_points(points)
        self.assertEqual(padded_points.shape[0], 2)
        self.assertEqual(padded_points.shape[1], 7)
        self.assertEqual(padded_points.shape[-1], points.shape[-1] - 1)
        self.assertTrue(torch.all(torch.isclose(points[:, 1:4], padded_points[valid_points_mask])))

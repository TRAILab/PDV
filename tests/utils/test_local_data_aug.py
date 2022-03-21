#!/usr/bin/env python

import unittest
import numpy as np
import math

from pcdet.datasets.augmentor.augmentor_utils import local_rotation_and_translation

class TestBoxTransformation(unittest.TestCase):
    def testNoTransform(self):
        box = np.array([[0,0,0,3,4,5,0]],np.float64)
        points = np.array([[0,0,0,0]],np.float64)
        gt_names = np.array(['Vehicle'])
        deterministic_ranges = {'X_AXIS': 0.0, 'Y_AXIS': 0.0, 'Z_AXIS': 0.0, "ROT_ANGLE": 0.0}
        box_after, points_after = local_rotation_and_translation(box,points,gt_names,deterministic_vals=deterministic_ranges)
        np.testing.assert_allclose(box, box_after, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(points, points_after, rtol=1e-7, atol=1e-7)

    def testRotation(self):
        box = np.array([[0,0,0,3,4,5,0]],np.float64)
        points = np.array([[0,0,0,0]],np.float64)
        gt_names = np.array(['Pedestrian'])
        deterministic_ranges = {'X_AXIS': 0.0, 'Y_AXIS': 0.0, 'Z_AXIS': 0.0, "ROT_ANGLE": math.pi/2}
        box_after, points_after = local_rotation_and_translation(box,points,gt_names,deterministic_vals=deterministic_ranges)
        box_after_truth = np.array([[0,0,0,3,4,5,math.pi/2]],np.float64)
        np.testing.assert_allclose(box_after_truth, box_after, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(points, points_after, rtol=1e-7, atol=1e-7)
    
    def testTranslation(self):
        box = np.array([[0,0,0,3,4,5,0]],np.float64)
        points = np.array([[0,0,0,0]],np.float64)
        gt_names = np.array(['Cyclist'])
        deterministic_ranges = {'X_AXIS': 1.2, 'Y_AXIS': 3.0, 'Z_AXIS': -1.4, "ROT_ANGLE": 0.0}
        box_after, points_after = local_rotation_and_translation(box,points,gt_names,deterministic_vals=deterministic_ranges)
        box_after_truth = np.array([[1.2,3.0,-1.4,3,4,5,0.0]],np.float64)
        np.testing.assert_allclose(box_after_truth, box_after, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(points, points_after, rtol=1e-7, atol=1e-7)


class TestOverallTransformation(unittest.TestCase):
    def testNoTransform(self):
        box = np.array([[0,0,0,3,4,5,0]],np.float64)
        points = np.array([[0,0,0,0],
                           [1,1,1,0],
                           [-1,-1,-1,0]],np.float64)
        gt_names = np.array(['Car'])
        deterministic_ranges = {'X_AXIS': 0.0, 'Y_AXIS': 0.0, 'Z_AXIS': 0.0, "ROT_ANGLE": 0.0}
        box_after, points_after = local_rotation_and_translation(box,points,gt_names,deterministic_vals=deterministic_ranges)
        np.testing.assert_allclose(box, box_after, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(points, points_after, rtol=1e-7, atol=1e-7)

    def testRotation(self):
        box = np.array([[0,0,0,3,4,5,0]],np.float64)
        points = np.array([[0,0,0,0],
                           [1,1,1,0],
                           [-1,-1,-1,0]],np.float64)
        gt_names = np.array(['Pedestrian'])
        deterministic_ranges = {'X_AXIS': 0.0, 'Y_AXIS': 0.0, 'Z_AXIS': 0.0, "ROT_ANGLE": math.pi/2}
        box_after, points_after = local_rotation_and_translation(box,points,gt_names,deterministic_vals=deterministic_ranges)
        box_after_truth = np.array([[0,0,0,3,4,5,math.pi/2]],np.float64)
        points_after_truth = np.array([[0,0,0,0],
                                 [-1,1,1,0],
                                 [1,-1,-1,0]],np.float64)
        np.testing.assert_allclose(box_after_truth, box_after, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(points_after_truth, points_after, rtol=1e-7, atol=1e-7)

    def testTranslation(self):
        box = np.array([[0,0,0,3,4,5,0]],np.float64)
        points = np.array([[0,0,0,0],
                           [1,1,1,0],
                           [-1,-1,-1,0]],np.float64)
        gt_names = np.array(['Cyclist'])
        deterministic_ranges = {'X_AXIS': 1.2, 'Y_AXIS': -1.3, 'Z_AXIS': 0.8, "ROT_ANGLE": 0.0}
        box_after, points_after = local_rotation_and_translation(box,points,gt_names,deterministic_vals=deterministic_ranges)
        box_after_truth = np.array([[1.2,-1.3,0.8,3,4,5,0.0]],np.float64)
        points_after_truth = np.array([[1.2,-1.3,0.8,0],
                                       [2.2,-0.3,1.8,0],
                                       [0.2,-2.3,-0.2,0]],np.float64)
        np.testing.assert_allclose(box_after_truth, box_after, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(points_after_truth, points_after, rtol=1e-7, atol=1e-7)

    def testOverall(self):
        box = np.array([[0,0,0,3,4,5,0]],np.float64)
        points = np.array([[0,0,0,0],
                           [1,1,1,0],
                           [-1,-1,-1,0]],np.float64)
        gt_names = np.array(['Vehicle'])
        deterministic_ranges = {'X_AXIS': 1.0, 'Y_AXIS': -1.0, 'Z_AXIS': 0.5, "ROT_ANGLE": math.pi/2}
        box_after, points_after = local_rotation_and_translation(box,points,gt_names,deterministic_vals=deterministic_ranges)
        box_after_truth = np.array([[1,-1,0.5,3,4,5,math.pi/2]],np.float64)
        points_after_truth = np.array([[1,-1,0.5,0],
                                       [0,0,1.5,0],
                                       [2,-2,-0.5,0]],np.float64)
        np.testing.assert_allclose(box_after_truth, box_after)
        np.testing.assert_allclose(points_after, points_after_truth, rtol=1e-7, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
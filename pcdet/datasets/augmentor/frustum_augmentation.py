import pickle

import numpy as np
import torch
from easydict import EasyDict

from ...ops.iou3d_nms import iou3d_nms_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils


class FrustumAugmentation:
    def __init__(self, frustum_cfg, class_names, logger=None):
        self.frustum_cfg = {}
        self.class_names = class_names
        self.logger = logger

        for class_name in self.class_names:
            self.frustum_cfg[class_name] = EasyDict(frustum_cfg.DEFAULT_PARAMS)
            if class_name.upper() in frustum_cfg.get('CLASS_PARAMS', {}):
                self.frustum_cfg[class_name].update(frustum_cfg.CLASS_PARAMS[class_name.upper()])

    def get_class_frustum_cfg(self, class_name):
        return self.frustum_cfg[class_name]

    def get_points_in_frustums_mask(self, data_dict):
        points = data_dict['points']
        gt_boxes = data_dict['gt_boxes']
        gt_names = data_dict['gt_names']

        point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
            torch.from_numpy(points[:, 0:3]),
            torch.from_numpy(gt_boxes)
        ).numpy()  # (nboxes, npoints)

        # Get points in the box
        corners3d = np.zeros((len(point_indices), 5, 3))
        frustum_masks = np.zeros((len(point_indices), points.shape[0]))
        for i, points_in_box_indices in enumerate(point_indices):
            frustum_cfg = self.get_class_frustum_cfg(gt_names[i])
            if points_in_box_indices.sum() < frustum_cfg.MIN_NUM_POINTS:
                continue

            # Choose random point in each ground truth box
            points_in_box = points[points_in_box_indices > 0]
            if points_in_box.shape[0] <= 0:
                continue
            point = points_in_box[np.random.choice(points_in_box.shape[0], 1)[0]]

            # Find points in frustum
            frustum_corners3d = box_utils.frustum_params_to_corners3d(point[:3], frustum_cfg)
            frustum_mask = box_utils.points_in_frustum(points[:, 0:3], frustum_corners3d)
            # Make sure frustum is only affecting points in current gt box
            frustum_mask = np.logical_and(frustum_mask, points_in_box_indices)
            frustum_masks[i] = frustum_mask
            # frustum_masks = np.bitwise_or(frustum_masks, frustum_mask)

            corners3d[i] = frustum_corners3d

        return frustum_masks, corners3d


class FrustumDropout(FrustumAugmentation):
    def __init__(self, frustum_cfg, class_names, logger=None):
        super().__init__(frustum_cfg, class_names, logger)

    def __call__(self, data_dict):
        # Get frustum mask
        frustum_masks, frustum_corners3d = self.get_points_in_frustums_mask(data_dict)

        # Check if dropout happens
        perform_dropout = np.zeros((frustum_masks.shape), dtype=bool)
        for i, frustum_mask in enumerate(frustum_masks):
            frustum_cfg = self.get_class_frustum_cfg(data_dict['gt_names'][i])
            prob = frustum_cfg.PROBABILITY
            prob_size = perform_dropout.shape[1] if frustum_cfg.get('INDIV_POINT_DROPOUT') else 1
            perform_dropout[i] = np.random.choice([True, False], p=[prob, 1-prob], size=(prob_size,))

        # Merge masks into one and filter data
        frustum_masks = np.logical_and(frustum_masks, perform_dropout)
        frustum_mask = frustum_masks.sum(axis=0)

        data_dict['points'] = data_dict['points'][frustum_mask == 0]
        # need to make sure if they exist, just append

        if 'frustums' in data_dict:
            data_dict['frustums'] = np.append(data_dict['frustums'], frustum_corners3d, axis=0)
        else:
            data_dict['frustums'] = frustum_corners3d
        return data_dict


class FrustumNoise(FrustumAugmentation):
    def __init__(self, frustum_cfg, class_names, logger=None):
        super().__init__(frustum_cfg, class_names, logger)

    def __call__(self, data_dict):
        # Get frustum mask
        frustum_masks, frustum_corners3d = self.get_points_in_frustums_mask(data_dict)

        # Set noise level
        noise = np.zeros((frustum_masks.shape[0], 3))
        for i, frustum_mask in enumerate(frustum_masks):
            frustum_cfg = self.get_class_frustum_cfg(data_dict['gt_names'][i])
            noise[i,:] = np.random.normal(0, frustum_cfg.MAX_NOISE_LEVEL, size=(3,))

        # Merge masks into one and filter data
        frustum_noise = frustum_masks.T @ noise

        data_dict['points'][:, :3] = data_dict['points'][:, :3] + frustum_noise
        # need to make sure if they exist, just append
        # data_dict['frustums'] = frustum_corners3d
        if 'frustums' in data_dict:
            data_dict['frustums'] = np.append(data_dict['frustums'], frustum_corners3d, axis=0)
        else:
            data_dict['frustums'] = frustum_corners3d
        return data_dict

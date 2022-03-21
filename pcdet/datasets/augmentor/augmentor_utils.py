import copy
import numpy as np
import torch

from ...utils import common_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...ops.iou3d_nms import iou3d_nms_utils


def random_flip_along_x(gt_boxes, points):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]

    return gt_boxes, points


def random_flip_along_y(gt_boxes, points):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]

    return gt_boxes, points


def global_rotation(gt_boxes, points, rot_range):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]

    return gt_boxes, points


def global_scaling(gt_boxes, points, scale_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    return gt_boxes, points

def random_image_flip_horizontal(image, depth_map, gt_boxes, calib):
    """
    Performs random horizontal flip augmentation
    Args:
        image: (H_image, W_image, 3), Image
        depth_map: (H_depth, W_depth), Depth map
        gt_boxes: (N, 7), 3D box labels in LiDAR coordinates [x, y, z, w, l, h, ry]
        calib: calibration.Calibration, Calibration object
    Returns:
        aug_image: (H_image, W_image, 3), Augmented image
        aug_depth_map: (H_depth, W_depth), Augmented depth map
        aug_gt_boxes: (N, 7), Augmented 3D box labels in LiDAR coordinates [x, y, z, w, l, h, ry]
    """
    # Randomly augment with 50% chance
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])

    if enable:
        # Flip images
        aug_image = np.fliplr(image)
        aug_depth_map = np.fliplr(depth_map)

        # Flip 3D gt_boxes by flipping the centroids in image space
        aug_gt_boxes = copy.copy(gt_boxes)
        locations = aug_gt_boxes[:, :3]
        img_pts, img_depth = calib.lidar_to_img(locations)
        W = image.shape[1]
        img_pts[:, 0] = W - img_pts[:, 0]
        pts_rect = calib.img_to_rect(u=img_pts[:, 0], v=img_pts[:, 1], depth_rect=img_depth)
        pts_lidar = calib.rect_to_lidar(pts_rect)
        aug_gt_boxes[:, :3] = pts_lidar
        aug_gt_boxes[:, 6] = -1 * aug_gt_boxes[:, 6]

    else:
        aug_image = image
        aug_depth_map = depth_map
        aug_gt_boxes = gt_boxes

    return aug_image, aug_depth_map, aug_gt_boxes

def local_point_dropout(gt_boxes, points, min_num_points, probability):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        min_num_points: int
        probability: float from 0-1
    Returns:
    """

    # Extract points in each ground truth box
    point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
        torch.from_numpy(points[:, 0:3]),
        torch.from_numpy(gt_boxes)
    ).numpy()
    # print(point_indices)
    num_points_in_boxes = point_indices.sum(axis=1)

    all_point_indices = point_indices[num_points_in_boxes >= min_num_points].sum(axis=0)

    dropout = np.random.choice([True, False], p=[probability, 1. - probability], size=(all_point_indices.shape))
    dropout = np.logical_and(dropout, all_point_indices)
    points = points[np.logical_not(dropout)]
    return points

def local_rotation_and_translation(gt_boxes, points, gt_names, rot_range=None, translation_range=None, disabled_classes=None, deterministic_vals=None):
    """
    Following the local data augmentations implemented in VoxelNet: https://arxiv.org/abs/1711.06396

    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
        translation_range: dict: {'X_AXIS': [mean, std], 'Y_AXIS': [mean, std], etc.}
        deterministic_vals: dict: {'ROT_ANGLE': angle, 'X_AXIS': t_x, 'Y_AXIS': t_y, etc.}
    Returns:
    """

    # Extract points in each ground truth box
    point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
        torch.from_numpy(points[:, 0:3]),
        torch.from_numpy(gt_boxes)
    ).numpy()

    translation = []
    translated_axes = []
    if deterministic_vals is not None:
        # deterministic_behaviours
        if deterministic_vals.get('X_AXIS', None) is not None:
            translated_axes.append(0)
            translation.append(deterministic_vals['X_AXIS'])
        if deterministic_vals.get('Y_AXIS', None) is not None:
            translated_axes.append(1)
            translation.append(deterministic_vals['Y_AXIS'])
        if deterministic_vals.get('Z_AXIS', None) is not None:
            translated_axes.append(2)
            translation.append(deterministic_vals['Z_AXIS'])
        if deterministic_vals.get('ROT_ANGLE', None) is not None:
            noise_rotation = deterministic_vals['ROT_ANGLE']
    else:
        # random behaviours
        if translation_range.get('X_AXIS', None) is not None:
            translated_axes.append(0)
            tx_range = translation_range['X_AXIS']
        if translation_range.get('Y_AXIS', None) is not None:
            translated_axes.append(1)
            ty_range = translation_range['Y_AXIS']
        if translation_range.get('Z_AXIS', None) is not None:
            translated_axes.append(2)
            tz_range = translation_range['Z_AXIS']

    is_collision = True
    for i in range(len(gt_boxes)):
        if disabled_classes is not None and gt_names[i] in disabled_classes:
            continue

        if deterministic_vals is None:
            # random rotation
            noise_rotation = np.random.uniform(rot_range[0], rot_range[1])

            # random translations
            translation = []
            if 0 in translated_axes:
                translation.append(np.random.normal(tx_range[0],tx_range[1]))
            if 1 in translated_axes:
                translation.append(np.random.normal(ty_range[0],ty_range[1]))
            if 2 in translated_axes:
                translation.append(np.random.normal(tz_range[0],tz_range[1]))

        translation = np.array(translation)

        # box rotation and translation
        tmp = copy.deepcopy(gt_boxes[i]).reshape(1,-1)
        tmp[:, 0:3] = common_utils.rotate_points_along_z(tmp[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
        tmp[:, translated_axes] += translation
        tmp[:, 6] += noise_rotation

        # check collision
        is_collision = False
        bev_ious = iou3d_nms_utils.boxes_bev_iou_cpu(tmp,np.delete(gt_boxes,i,0))
        is_collision = not np.all((bev_ious == 0.0))

        if not is_collision:
            points_inside_box = point_indices[i].astype(bool)
            points[points_inside_box] = common_utils.rotate_points_along_z(points[np.newaxis, points_inside_box, :], np.array([noise_rotation]))[0]
            gt_boxes[i] = tmp
            transformed_points = copy.deepcopy(points[points_inside_box])
            transformed_points[:,[translated_axes]] += translation
            points[points_inside_box] = transformed_points

    return gt_boxes, points

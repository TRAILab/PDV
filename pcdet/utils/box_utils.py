import numpy as np
import scipy
import torch
import copy
from scipy.spatial import Delaunay

from ..ops.roiaware_pool3d import roiaware_pool3d_utils
from . import common_utils


def in_hull(p, hull):
    """
    :param p: (N, K) test points
    :param hull: (M, K) M corners of a box
    :return (N) bool
    """
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)

    return flag


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = common_utils.rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def frustum_params_to_corners3d(point, frustum_params):
    """
    3 ------ 0
    | \     /|
    |  \   / |
    |   \ /  |
    |    4   |
    |   / \  |
    |  /   \ |
    | /     \|
    2 ------ 1
    Args:
        point: point frustum is looking at (3,) [x, y, z]

        frustum params:
            theta: width FOV (rad)
            phi: height FOV (rad)
            dist: length of frustum

    Returns:
        points: (5, 3) [x, y, z]
    """
    point_unit = point / np.linalg.norm(point)
    look_from = point - frustum_params.DISTANCE_TO_POINT/2 * point_unit
    look_at = point + frustum_params.DISTANCE_TO_POINT/2 * point_unit
    up_vec = np.array([0., 0., 1])

    v_x = look_at - look_from
    v_x = v_x / np.linalg.norm(v_x)

    v_y = np.cross(up_vec, v_x)
    v_y = v_y / np.linalg.norm(v_y)

    v_z = np.cross(v_x, v_y)

    f_width = 2*np.linalg.norm(look_at - look_from) * np.tan(frustum_params.FOV_WIDTH/2)
    f_height = 2*np.linalg.norm(look_at - look_from) * np.tan(frustum_params.FOV_HEIGHT/2)

    p0 = look_at - f_width/2 * v_y + f_height/2 * v_z
    p1 = look_at - f_width/2 * v_y - f_height/2 * v_z
    p2 = look_at + f_width/2 * v_y - f_height/2 * v_z
    p3 = look_at + f_width/2 * v_y + f_height/2 * v_z
    p4 = look_from

    points = np.vstack((p0, p1, p2, p3, p4))

    return points


def points_in_frustum(points, frustum_corners3d):
    """
    3 ------ 0
    | \     /|
    |  \   / |
    |   \ /  |
    |    4   |
    |   / \  |
    |  /   \ |
    | /     \|
    2 ------ 1
    Frustum vectors for each plane to define normal:
        4,3 x 4,0
        4,0 x 4,1
        4,1 x 4,2
        4,2,x 4,3
        1,0 x 1,2
    Args:
        points: pointcloud (N, 3) [x, y, z]

        frustum_corners3d: Frustum point corners (as in diagram) (5, 3)

    Returns:
        points_in_frustum: Mask for which points are in frustum (N,)
    """
    frustum_planes_vecs_idxs = np.array([
        [[4, 3], [4, 0]],
        [[4, 0], [4, 1]],
        [[4, 1], [4, 2]],
        [[4, 2], [4, 3]],
        [[1, 0], [1, 2]]
    ])

    point_idx_on_planes = np.array([4, 4, 4, 4, 0])

    points_in_frustum = np.ones((points.shape[0]), dtype=bool)
    for i, plane_vecs_idxs in enumerate(frustum_planes_vecs_idxs):
        vec1 = frustum_corners3d[plane_vecs_idxs[0][1]] - frustum_corners3d[plane_vecs_idxs[0][0]]
        vec2 = frustum_corners3d[plane_vecs_idxs[1][1]] - frustum_corners3d[plane_vecs_idxs[1][0]]
        plane_normal = np.cross(vec1, vec2)

        # Halfspace check
        vec_test = points - frustum_corners3d[point_idx_on_planes[i]]
        dist = np.dot(vec_test, plane_normal)
        halfspace_mask = dist > 0

        # Keep track of points that are within all halfspaces
        points_in_frustum = np.bitwise_and(points_in_frustum, halfspace_mask)

    return points_in_frustum


def mask_boxes_outside_range_numpy(boxes, limit_range, min_num_corners=1):
    """
    Args:
        boxes: (N, 7) [x, y, z, dx, dy, dz, heading, ...], (x, y, z) is the box center
        limit_range: [minx, miny, minz, maxx, maxy, maxz]
        min_num_corners:

    Returns:

    """
    if boxes.shape[1] > 7:
        boxes = boxes[:, 0:7]
    corners = boxes_to_corners_3d(boxes)  # (N, 8, 3)
    mask = ((corners >= limit_range[0:3]) & (corners <= limit_range[3:6])).all(axis=2)
    mask = mask.sum(axis=1) >= min_num_corners  # (N)

    return mask


def remove_points_in_boxes3d(points, boxes3d):
    """
    Args:
        points: (num_points, 3 + C)
        boxes3d: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps

    Returns:

    """
    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)
    points, is_numpy = common_utils.check_numpy_to_torch(points)
    point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, 0:3], boxes3d)
    points = points[point_masks.sum(dim=0) == 0]

    return points.numpy() if is_numpy else points


def boxes3d_kitti_camera_to_lidar(boxes3d_camera, calib):
    """
    Args:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
        calib:

    Returns:
        boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    """
    boxes3d_camera_copy = copy.deepcopy(boxes3d_camera)
    xyz_camera, r = boxes3d_camera_copy[:, 0:3], boxes3d_camera_copy[:, 6:7]
    l, h, w = boxes3d_camera_copy[:, 3:4], boxes3d_camera_copy[:, 4:5], boxes3d_camera_copy[:, 5:6]

    xyz_lidar = calib.rect_to_lidar(xyz_camera)
    xyz_lidar[:, 2] += h[:, 0] / 2
    return np.concatenate([xyz_lidar, l, w, h, -(r + np.pi / 2)], axis=-1)


def boxes3d_kitti_fakelidar_to_lidar(boxes3d_lidar):
    """
    Args:
        boxes3d_fakelidar: (N, 7) [x, y, z, w, l, h, r] in old LiDAR coordinates, z is bottom center

    Returns:
        boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    """
    boxes3d_lidar_copy = copy.deepcopy(boxes3d_lidar)
    w, l, h = boxes3d_lidar_copy[:, 3:4], boxes3d_lidar_copy[:, 4:5], boxes3d_lidar_copy[:, 5:6]
    r = boxes3d_lidar_copy[:, 6:7]

    boxes3d_lidar_copy[:, 2] += h[:, 0] / 2
    return np.concatenate([boxes3d_lidar_copy[:, 0:3], l, w, h, -(r + np.pi / 2)], axis=-1)


def boxes3d_kitti_lidar_to_fakelidar(boxes3d_lidar):
    """
    Args:
        boxes3d_lidar: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
        boxes3d_fakelidar: [x, y, z, w, l, h, r] in old LiDAR coordinates, z is bottom center

    """
    boxes3d_lidar_copy = copy.deepcopy(boxes3d_lidar)
    dx, dy, dz = boxes3d_lidar_copy[:, 3:4], boxes3d_lidar_copy[:, 4:5], boxes3d_lidar_copy[:, 5:6]
    heading = boxes3d_lidar_copy[:, 6:7]

    boxes3d_lidar_copy[:, 2] -= dz[:, 0] / 2
    return np.concatenate([boxes3d_lidar_copy[:, 0:3], dy, dx, dz, -heading - np.pi / 2], axis=-1)


def enlarge_box3d(boxes3d, extra_width=(0, 0, 0)):
    """
    Args:
        boxes3d: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
        extra_width: [extra_x, extra_y, extra_z]

    Returns:

    """
    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)
    large_boxes3d = boxes3d.clone()

    large_boxes3d[:, 3:6] += boxes3d.new_tensor(extra_width)[None, :]
    return large_boxes3d


def boxes3d_lidar_to_kitti_camera(boxes3d_lidar, calib):
    """
    :param boxes3d_lidar: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    :param calib:
    :return:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
    """
    boxes3d_lidar_copy = copy.deepcopy(boxes3d_lidar)
    xyz_lidar = boxes3d_lidar_copy[:, 0:3]
    l, w, h = boxes3d_lidar_copy[:, 3:4], boxes3d_lidar_copy[:, 4:5], boxes3d_lidar_copy[:, 5:6]
    r = boxes3d_lidar_copy[:, 6:7]

    xyz_lidar[:, 2] -= h.reshape(-1) / 2
    xyz_cam = calib.lidar_to_rect(xyz_lidar)
    # xyz_cam[:, 1] += h.reshape(-1) / 2
    r = -r - np.pi / 2
    return np.concatenate([xyz_cam, l, h, w, r], axis=-1)


def boxes3d_to_corners3d_kitti_camera(boxes3d, bottom_center=True):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, ry] in camera coords, see the definition of ry in KITTI dataset
    :param bottom_center: whether y is on the bottom center of object
    :return: corners3d: (N, 8, 3)
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    boxes_num = boxes3d.shape[0]
    l, h, w = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2], dtype=np.float32).T
    z_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T
    if bottom_center:
        y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        y_corners[:, 4:8] = -h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)
    else:
        y_corners = np.array([h / 2., h / 2., h / 2., h / 2., -h / 2., -h / 2., -h / 2., -h / 2.], dtype=np.float32).T

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                         [zeros, ones, zeros],
                         [np.sin(ry), zeros, np.cos(ry)]])  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                   z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)
    rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
    x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

    return corners.astype(np.float32)


def boxes3d_kitti_camera_to_imageboxes(boxes3d, calib, image_shape=None):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
    :param calib:
    :return:
        box_2d_preds: (N, 4) [x1, y1, x2, y2]
    """
    corners3d = boxes3d_to_corners3d_kitti_camera(boxes3d)
    pts_img, _ = calib.rect_to_img(corners3d.reshape(-1, 3))
    corners_in_image = pts_img.reshape(-1, 8, 2)

    min_uv = np.min(corners_in_image, axis=1)  # (N, 2)
    max_uv = np.max(corners_in_image, axis=1)  # (N, 2)
    boxes2d_image = np.concatenate([min_uv, max_uv], axis=1)
    if image_shape is not None:
        boxes2d_image[:, 0] = np.clip(boxes2d_image[:, 0], a_min=0, a_max=image_shape[1] - 1)
        boxes2d_image[:, 1] = np.clip(boxes2d_image[:, 1], a_min=0, a_max=image_shape[0] - 1)
        boxes2d_image[:, 2] = np.clip(boxes2d_image[:, 2], a_min=0, a_max=image_shape[1] - 1)
        boxes2d_image[:, 3] = np.clip(boxes2d_image[:, 3], a_min=0, a_max=image_shape[0] - 1)

    return boxes2d_image


def boxes_iou_normal(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 4) [x1, y1, x2, y2]
        boxes_b: (M, 4) [x1, y1, x2, y2]

    Returns:

    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 4
    x_min = torch.max(boxes_a[:, 0, None], boxes_b[None, :, 0])
    x_max = torch.min(boxes_a[:, 2, None], boxes_b[None, :, 2])
    y_min = torch.max(boxes_a[:, 1, None], boxes_b[None, :, 1])
    y_max = torch.min(boxes_a[:, 3, None], boxes_b[None, :, 3])
    x_len = torch.clamp_min(x_max - x_min, min=0)
    y_len = torch.clamp_min(y_max - y_min, min=0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    a_intersect_b = x_len * y_len
    iou = a_intersect_b / torch.clamp_min(area_a[:, None] + area_b[None, :] - a_intersect_b, min=1e-6)
    return iou


def boxes3d_lidar_to_aligned_bev_boxes(boxes3d):
    """
    Args:
        boxes3d: (N, 7 + C) [x, y, z, dx, dy, dz, heading] in lidar coordinate

    Returns:
        aligned_bev_boxes: (N, 4) [x1, y1, x2, y2] in the above lidar coordinate
    """
    rot_angle = common_utils.limit_period(boxes3d[:, 6], offset=0.5, period=np.pi).abs()
    choose_dims = torch.where(rot_angle[:, None] < np.pi / 4, boxes3d[:, [3, 4]], boxes3d[:, [4, 3]])
    aligned_bev_boxes = torch.cat((boxes3d[:, 0:2] - choose_dims / 2, boxes3d[:, 0:2] + choose_dims / 2), dim=1)
    return aligned_bev_boxes


def boxes3d_nearest_bev_iou(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:

    """
    boxes_bev_a = boxes3d_lidar_to_aligned_bev_boxes(boxes_a)
    boxes_bev_b = boxes3d_lidar_to_aligned_bev_boxes(boxes_b)

    return boxes_iou_normal(boxes_bev_a, boxes_bev_b)

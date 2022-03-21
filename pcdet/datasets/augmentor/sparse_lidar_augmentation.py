import numpy as np


def pto_ang_map(velo_points, H=64, W=512, slice=1):
    """
    Code from PseudoLiDAR++: https://github.com/mileyan/Pseudo_Lidar_V2
    :param H: the row num of depth map, could be 64(default), 32, 16
    :param W: the col num of depth map
    :param slice: output every slice lines
    """

    dtheta = np.radians(0.4 * 64.0 / H)
    dphi = np.radians(90.0 / W)

    x, y, z, i = velo_points[:, 0], velo_points[:, 1], velo_points[:, 2], velo_points[:, 3]

    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = np.sqrt(x ** 2 + y ** 2)
    d[d == 0] = 0.000001
    r[r == 0] = 0.000001
    phi = np.radians(45.) - np.arcsin(y / r)
    phi_ = (phi / dphi).astype(int)
    phi_[phi_ < 0] = 0
    phi_[phi_ >= W] = W - 1

    theta = np.radians(2.) - np.arcsin(z / d)
    theta_ = (theta / dtheta).astype(int)
    theta_[theta_ < 0] = 0
    theta_[theta_ >= H] = H - 1

    depth_map = - np.ones((H, W, 4))
    depth_map[theta_, phi_, 0] = x
    depth_map[theta_, phi_, 1] = y
    depth_map[theta_, phi_, 2] = z
    depth_map[theta_, phi_, 3] = i
    depth_map = depth_map[0::slice, 0::slice, :]
    depth_map = depth_map.reshape((-1, 4))
    depth_map = depth_map[depth_map[:, 0] != -1.0]
    return depth_map

import torch


def get_padded_points(points, point_features=None):
    """
    Original points_in_boxes_gpu does not accept uneven amounts of data. This is a wrapper function to pad points
    so point batch is the same size.

    Args:
        points: (N1 + N2 + ... + NB, bxyz)
        point_features [optional]: (N1 + N2 + ... + NB, f)
    Returns:
        padded_points: (B, N, xyz) where N is the max number for points per batch or max(N1, N2, ..., NB)
        point_features [optional]: (B, N, f)
        valid_points_mask: (B, N)
    """
    batch_idx = points[:, 0]
    batch_ids, xyz_batch_cnt = batch_idx.unique(return_counts=True)
    batch_size = batch_ids.shape[0]

    max_xyz_batch_cnt = xyz_batch_cnt.max()

    # TODO: Look for way to fix the number of points
    padded_points = torch.zeros((batch_size, max_xyz_batch_cnt, points.shape[1]-1), dtype=points.dtype, device=points.device)
    if point_features is not None:
        padded_point_features = torch.zeros((batch_size, max_xyz_batch_cnt, point_features.shape[1]), dtype=point_features.dtype, device=point_features.device)
    valid_points_mask = torch.zeros(batch_size, max_xyz_batch_cnt, dtype=torch.bool, device=points.device)
    index = 0
    for k in range(batch_size):
        padded_points[k, :xyz_batch_cnt[k], :] = points[index:index + xyz_batch_cnt[k], 1:]
        if point_features is not None:
            padded_point_features[k, :xyz_batch_cnt[k], :] = point_features[index:index + xyz_batch_cnt[k], :]
        valid_points_mask[k, :xyz_batch_cnt[k]] = True
        index += xyz_batch_cnt[k]

    if point_features is not None:
        return padded_points, padded_point_features, valid_points_mask
    return padded_points, valid_points_mask

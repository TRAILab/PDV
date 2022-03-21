import torch

from .vfe_template import VFETemplate
from ....ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu

class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        batch_dict['voxel_features'] = points_mean.contiguous()

        return batch_dict


class MeanOrientationVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features + 1

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        orientation_features = torch.zeros(points_mean.shape[0], device=points_mean.device)
        _, counts = batch_dict['voxel_coords'][:, 0].unique(return_counts=True)
        centroids = torch.zeros(batch_dict['batch_size'], counts.max(), 3, device=points_mean.device)
        cur_idx = 0
        for i, count in enumerate(counts):
            centroids[i, :count] = points_mean[cur_idx:cur_idx + count, 0:3]
            cur_idx += count
        centroids_in_gt_boxes = points_in_boxes_gpu(centroids, batch_dict['gt_boxes'][..., 0:7])
        gt_box_indices = torch.zeros(points_mean.shape[0], dtype=torch.int64, device=points_mean.device)
        cur_idx = 0
        for i, count in enumerate(counts):
            gt_box_indices[cur_idx:cur_idx + count] = centroids_in_gt_boxes[i, :count]
        gt_box_indices_valid_mask = gt_box_indices != -1
        orientation_features[gt_box_indices_valid_mask] = batch_dict['gt_boxes'][batch_dict['voxel_coords'][:, 0].long()[gt_box_indices_valid_mask],
                                                                                 gt_box_indices[gt_box_indices_valid_mask],
                                                                                 6]
        points_mean = torch.cat((points_mean, orientation_features.unsqueeze(1)), dim=-1)
        batch_dict['voxel_features'] = points_mean.contiguous()

        return batch_dict

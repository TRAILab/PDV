# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.

import os
import pickle
import copy
import numpy as np
import torch
import multiprocessing
from tqdm import tqdm
from pathlib import Path
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate


class WaymoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.data_path = self.root_path / self.dataset_cfg.PROCESSED_DATA_TAG
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]

        self.infos = []
        self.densification_infos = []
        self.include_waymo_data(self.mode)

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        self.infos = []
        self.include_waymo_data(self.mode)

    def include_waymo_data(self, mode):
        self.logger.info('Loading Waymo dataset')
        waymo_infos = []

        num_skipped_infos = 0
        for k in range(len(self.sample_sequence_list)):
            sequence_name = os.path.splitext(self.sample_sequence_list[k])[0]
            info_path = self.data_path / sequence_name / ('%s.pkl' % sequence_name)
            info_path = self.check_sequence_name_with_all_version(info_path)
            if not info_path.exists():
                num_skipped_infos += 1
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                waymo_infos.extend(infos)

        self.infos.extend(waymo_infos[:])
        self.logger.info('Total skipped info %s' % num_skipped_infos)
        self.logger.info('Total samples for Waymo dataset: %d' % (len(waymo_infos)))

        if self.dataset_cfg.SAMPLED_INTERVAL[mode] > 1:
            sampled_waymo_infos = []
            for k in range(0, len(self.infos), self.dataset_cfg.SAMPLED_INTERVAL[mode]):
                sampled_waymo_infos.append(self.infos[k])
            self.infos = sampled_waymo_infos
            self.logger.info('Total sampled samples for Waymo dataset: %d' % len(self.infos))
        
        if self.dataset_cfg.get('NUMBER_OF_SWEEPS_USED', 1) > 1:
            # Check if it is for test set
            processed_data_tag_test_set = self.dataset_cfg.get('PROCESSED_DATA_TAG_TEST_SET', 'waymo_processed_data_test_set')
            if self.data_path == self.root_path / processed_data_tag_test_set:
                densification_infos_file = self.root_path / ('densification_infos_test.pkl')
                if densification_infos_file.exists():
                    self.densification_infos = pickle.load(open(densification_infos_file, 'rb'))
                else:
                    print('densification infos test file not found')
            else:
                densification_infos_file = self.root_path / ('densification_infos_%s.pkl' %self.split)
                if densification_infos_file.exists():
                    self.densification_infos = pickle.load(open(densification_infos_file, 'rb'))
                else:
                    print('densification infos file not found')

    @staticmethod
    def check_sequence_name_with_all_version(sequence_file):
        if '_with_camera_labels' not in str(sequence_file) and not sequence_file.exists():
            sequence_file = Path(str(sequence_file)[:-9] + '_with_camera_labels.tfrecord')
        if '_with_camera_labels' in str(sequence_file) and not sequence_file.exists():
            sequence_file = Path(str(sequence_file).replace('_with_camera_labels', ''))

        return sequence_file

    def get_infos(self, raw_data_path, save_path, num_workers=multiprocessing.cpu_count(), has_label=True, sampled_interval=1, both_returns=False, test_set=False):
        import concurrent.futures as futures
        from functools import partial
        from . import waymo_utils
        print('---------------The waymo sample interval is %d, total sequences is %d-----------------'
              % (sampled_interval, len(self.sample_sequence_list)))

        process_single_sequence = partial(
            waymo_utils.process_single_sequence,
            save_path=save_path, sampled_interval=sampled_interval, has_label=has_label, dataset_cfg=self.dataset_cfg, both_returns=both_returns, test_set=test_set
        )
        sample_sequence_file_list = [
            self.check_sequence_name_with_all_version(raw_data_path / sequence_file)
            for sequence_file in self.sample_sequence_list
        ]

        # process_single_sequence(sample_sequence_file_list[0])
        with futures.ThreadPoolExecutor(num_workers) as executor:
            sequence_infos = list(tqdm(executor.map(process_single_sequence, sample_sequence_file_list),
                                       total=len(sample_sequence_file_list)))
        all_sequences_infos = [item for infos in sequence_infos for item in infos]
        return all_sequences_infos


    def get_densification_infos(self, raw_data_path, num_workers=multiprocessing.cpu_count(), sampled_interval=1, max_sweeps=0):
        import concurrent.futures as futures
        from functools import partial
        from . import waymo_utils
        print('---------------The waymo sample interval is %d, total sequecnes is %d-----------------'
            % (sampled_interval, len(self.sample_sequence_list)))

        process_single_sequence = partial(
            waymo_utils.extract_densification_infos, sampled_interval=sampled_interval, max_sweeps=max_sweeps)
        sample_sequence_file_list = [
            self.check_sequence_name_with_all_version(raw_data_path / sequence_file)
            for sequence_file in self.sample_sequence_list
        ]

        with futures.ThreadPoolExecutor(num_workers) as executor:
            sequence_infos = list(tqdm(executor.map(process_single_sequence, sample_sequence_file_list),
                                    total=len(sample_sequence_file_list)))
        all_sequences_infos = [item for infos in sequence_infos for item in infos]
        return all_sequences_infos


    def get_name_time(self, raw_data_path, num_workers=multiprocessing.cpu_count(), sampled_interval=1):
        import concurrent.futures as futures
        from functools import partial
        from . import waymo_utils
        print('---------------The waymo sample interval is %d, total sequences is %d-----------------'
            % (sampled_interval, len(self.sample_sequence_list)))

        process_single_sequence = partial(
            waymo_utils.extract_name_time, sampled_interval=sampled_interval)
        sample_sequence_file_list = [
            self.check_sequence_name_with_all_version(raw_data_path / sequence_file)
            for sequence_file in self.sample_sequence_list
        ]

        with futures.ThreadPoolExecutor(num_workers) as executor:
            sequence_infos = list(tqdm(executor.map(process_single_sequence, sample_sequence_file_list),
                                    total=len(sample_sequence_file_list)))
        all_sequences_infos = [item for infos in sequence_infos for item in infos]
        return all_sequences_infos

    def get_lidar(self, sequence_name, sample_idx, both_returns=False):
        lidar_file = self.data_path / sequence_name / ('%04d.npy' % sample_idx)
        point_features = np.load(lidar_file)  # (N, 7): [x, y, z, intensity, elongation, NLZ_flag]

        points_all, NLZ_flag = point_features[:, 0:5], point_features[:, 5]
        points_all = points_all[NLZ_flag == -1]
        points_all[:, 3] = np.tanh(points_all[:, 3])
        if both_returns:
            points_2nd_ret = self.get_lidar_2nd_ret(sequence_name, sample_idx)
            points_all = np.concatenate([points_all, points_2nd_ret], axis=0)
        return points_all

    def get_lidar_2nd_ret(self, sequence_name, sample_idx):
        # Check if it is for test set
        processed_data_tag_test_set = self.dataset_cfg.get('PROCESSED_DATA_TAG_TEST_SET', 'waymo_processed_data_test_set')
        if self.data_path == self.root_path / processed_data_tag_test_set:
            second_return_tag = self.dataset_cfg.get('SECOND_RETURN_PROCESSED_DATA_TAG_TEST_SET','waymo_processed_data_second_return_test_set')
        else:
            second_return_tag = self.dataset_cfg.get('SECOND_RETURN_PROCESSED_DATA_TAG','waymo_processed_data_second_return')

        lidar_file = self.data_path.parent / second_return_tag / sequence_name / ('%04d.npy' % sample_idx)
        point_features = np.load(lidar_file)  # (N, 7): [x, y, z, intensity, elongation, NLZ_flag]

        points_all, NLZ_flag = point_features[:, 0:5], point_features[:, 5]
        points_all = points_all[NLZ_flag == -1]
        points_all[:, 3] = np.tanh(points_all[:, 3])
        return points_all

    def get_sweep(self, sweep_info, both_returns=False):
        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        points_sweep = self.get_lidar(sweep_info['lidar_path'],sweep_info['sample_idx'], both_returns=both_returns)

        points_sweep = remove_ego_points(points_sweep).T
        if sweep_info['transform_matrix'] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

        cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T

    def get_lidar_with_sweeps(self, sequence_name, sample_idx, index, nsweeps=1, both_returns=False):

        points_all = self.get_lidar(sequence_name, sample_idx, both_returns=both_returns)

        sweep_points_list = [points_all]
        sweep_times_list = [np.zeros((points_all.shape[0], 1))]

        for k in np.random.choice(len(self.densification_infos[index]['sweeps']), nsweeps - 1, replace=False):
            points_sweep, times_sweep = self.get_sweep(self.densification_infos[index]['sweeps'][k])
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        densified_points = np.concatenate((points, times), axis=1)
        return densified_points

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']

        if self.dataset_cfg.get('SECOND_RETURN', False):
            if self.mode == 'train' and self.dataset_cfg.SECOND_RETURN.ENABLED_TRAINING:
                enable_second_ret = True
            elif self.mode == 'test' and self.dataset_cfg.SECOND_RETURN.ENABLED_TESTING:
                enable_second_ret = True
        else:
            enable_second_ret = False

        if self.dataset_cfg.get('NUMBER_OF_SWEEPS_USED', 1) > 1:
            points = self.get_lidar_with_sweeps(sequence_name, sample_idx, index, self.dataset_cfg.NUMBER_OF_SWEEPS_USED, both_returns=enable_second_ret)
        else:
            points = self.get_lidar(sequence_name, sample_idx, both_returns=enable_second_ret)

        input_dict = {
            'points': points,
            'frame_id': info['frame_id'],
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='unknown')

            if self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False):
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(annos['gt_boxes_lidar'])
            else:
                gt_boxes_lidar = annos['gt_boxes_lidar']

            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': gt_boxes_lidar,
                'num_points_in_gt': annos.get('num_points_in_gt', None)
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict['metadata'] = info.get('metadata', info['frame_id'])
        data_dict.pop('num_points_in_gt', None)
        return data_dict

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            map_name_to_kitti = {
                'Vehicle': 'Car',
                'Pedestrian': 'Pedestrian',
                'Cyclist': 'Cyclist',
                'Sign': 'Sign',
                'Car': 'Car'
            }
            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        def waymo_eval(eval_det_annos, eval_gt_annos, eval_levels_list_cfg=None):
            from .waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
            eval = OpenPCDetWaymoDetectionMetricsEstimator()

            # Overall Evaluation
            ap_dict = eval.waymo_evaluation(
                eval_det_annos, eval_gt_annos, class_name=class_names,
                distance_thresh=1000, fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            ap_result_str = '\n'
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0]
                ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

            # Evaluation across range
            if eval_levels_list_cfg:
                ap_result_str_list, ap_dict_list = [], []
                ap_result_str_list.append(ap_result_str)
                ap_dict_list.append(ap_dict)

                for i in range(len(eval_levels_list_cfg['RANGE_LIST'])-1):
                    lower_bound, upper_bound = eval_levels_list_cfg['RANGE_LIST'][i], eval_levels_list_cfg['RANGE_LIST'][i+1]

                    ap_dict = eval.waymo_evaluation(
                        eval_det_annos, eval_gt_annos, class_name=class_names,
                        distance_thresh=upper_bound, lower_bound=lower_bound, fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
                    )
                    ap_result_str = '\n'
                    for key in ap_dict:
                        ap_dict[key] = ap_dict[key][0]
                        ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])
                    
                    ap_result_str_list.append(ap_result_str)
                    ap_dict_list.append(ap_dict)

                return ap_result_str_list, ap_dict_list
            
            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos)
        elif kwargs['eval_metric'] == 'waymo':
            ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos, kwargs.get('eval_levels_list_cfg',None))
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict

    def create_groundtruth_database(self, info_path, save_path, used_classes=None, split='train', sampled_interval=10,
                                    processed_data_tag=None, densified_pc=False, both_returns=False):
        database_name = '%s_sampled_%d' % (split, sampled_interval)
        if densified_pc:
            database_name += '_densified'
        if both_returns:
            database_name += '_2_returns'
        database_save_path = save_path / ('pcdet_gt_database_' + database_name)
        db_info_save_path = save_path / ('pcdet_waymo_dbinfos_' + database_name + '.pkl')

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(0, len(infos), sampled_interval):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]

            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']

            if densified_pc:
                points = self.get_lidar_with_sweeps(sequence_name, sample_idx, k, nsweeps=self.dataset_cfg.NUMBER_OF_SWEEPS_USED, both_returns=both_returns)
            else:
                points = self.get_lidar(sequence_name, sample_idx, both_returns=both_returns)

            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            for i in range(num_obj):
                filename = '%s_%04d_%s_%d.bin' % (sequence_name, sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]
                gt_points[:, :3] -= gt_boxes[i, :3]

                if (used_classes is None) or names[i] in used_classes:
                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)

                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'sequence_name': sequence_name,
                               'sample_idx': sample_idx, 'gt_idx': i, 'box3d_lidar': gt_boxes[i],
                               'num_points_in_gt': gt_points.shape[0], 'difficulty': difficulty[i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)


def create_waymo_infos(dataset_cfg, class_names, data_path, save_path,
                       raw_data_tag='raw_data', processed_data_tag='waymo_processed_data',
                       sampled_interval=1, sampled_interval_gt=10, max_sweeps=0, densified_pc=False, both_returns=False,
                       workers=multiprocessing.cpu_count()):
    dataset = WaymoDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('waymo_infos_%s.pkl' % train_split)
    val_filename = save_path / ('waymo_infos_%s.pkl' % val_split)
    densification_train_filename = save_path / ('densification_infos_%s.pkl' % train_split)
    densification_val_filename = save_path / ('densification_infos_%s.pkl' % val_split)

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    waymo_infos_train = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=sampled_interval, both_returns=both_returns
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(waymo_infos_train, f)
    print('----------------Waymo info train file is saved to %s----------------' % train_filename)

    dataset.set_split(val_split)
    waymo_infos_val = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=sampled_interval, both_returns=both_returns
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(waymo_infos_val, f)
    print('----------------Waymo info val file is saved to %s----------------' % val_filename)

    if densified_pc and max_sweeps > 0:
        densification_val_filename = save_path / ('densification_infos_%s.pkl' % val_split)
        if not densification_val_filename.exists():
            densification_infos_val = dataset.get_densification_infos(
                raw_data_path=data_path / raw_data_tag,
                num_workers=workers, sampled_interval=sampled_interval, max_sweeps=max_sweeps
            )
            with open(densification_val_filename, 'wb') as f:
                pickle.dump(densification_infos_val, f)
            print('----------------Waymo densification info val file is saved to %s----------------' % densification_val_filename)

        dataset.set_split(train_split)
        densification_train_filename = save_path / ('densification_infos_%s.pkl' % train_split)
        if not densification_train_filename.exists():
            densification_infos_train = dataset.get_densification_infos(
                raw_data_path=data_path / raw_data_tag,
                num_workers=workers, sampled_interval=sampled_interval, max_sweeps=max_sweeps
            )
            with open(densification_train_filename, 'wb') as f:
                pickle.dump(densification_infos_train, f)
            print('----------------Waymo densification info train file is saved to %s----------------' % densification_train_filename)

    # Save infos for online server validation set submission
    val_time_name_filename = save_path / ('waymo_time_name_infos_%s.pkl' % val_split)
    if not val_time_name_filename.exists():
        waymo_name_time_infos_val = dataset.get_name_time(
            raw_data_path=data_path / raw_data_tag,
            num_workers=workers, sampled_interval=sampled_interval
        )
        with open(val_time_name_filename, 'wb') as f:
            pickle.dump(waymo_name_time_infos_val, f)
        print('----------------Waymo time name info val file is saved to %s----------------' % val_time_name_filename)


    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(
        info_path=train_filename, save_path=save_path, split='train', sampled_interval=sampled_interval_gt,
        used_classes=['Vehicle', 'Pedestrian', 'Cyclist'], densified_pc=densified_pc, both_returns=both_returns
    )
    print('---------------Data preparation Done---------------')

def create_waymo_test_set_infos(dataset_cfg, class_names,data_path, save_path, raw_data_tag='raw_data_test_set', 
                       processed_data_tag='waymo_processed_data_test_set',
                       sampled_interval=1, max_sweeps=0, densified_pc=False, both_returns=False, workers=multiprocessing.cpu_count()):
    dataset = WaymoDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )

    # Modify data path, sequence_list etc. to test set
    set_split_file = data_path / 'ImageSets' / 'test.txt'
    dataset.sample_sequence_list = [x.strip() for x in open(set_split_file).readlines()]
    dataset.data_path = data_path / processed_data_tag
    dataset.infos = []
    dataset.include_waymo_data('test')

    test_filename = save_path / 'waymo_infos_test.pkl'

    print('---------------Start to generate data infos---------------')
    waymo_infos_test = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=False,
        sampled_interval=sampled_interval, both_returns=both_returns, test_set=True
    )
    with open(test_filename, 'wb') as f:
        pickle.dump(waymo_infos_test, f)
    print('----------------Waymo info test file is saved to %s----------------' % test_filename)

    # Save infos for online server test set submission
    test_time_name_filename = save_path / 'waymo_time_name_infos_test.pkl'
    if not test_time_name_filename.exists():
        waymo_name_time_infos_val = dataset.get_name_time(
            raw_data_path=data_path / raw_data_tag,
            num_workers=workers, sampled_interval=sampled_interval
        )
        with open(test_time_name_filename, 'wb') as f:
            pickle.dump(waymo_name_time_infos_val, f)

        print('----------------Waymo time name info test file is saved to %s----------------' % test_time_name_filename)

    if densified_pc and max_sweeps > 0:
        densification_test_filename = save_path / 'densification_infos_test.pkl'
        if not densification_test_filename.exists():
            densification_infos_test = dataset.get_densification_infos(
                raw_data_path=data_path / raw_data_tag,
                num_workers=workers, sampled_interval=sampled_interval, max_sweeps=max_sweeps
            )
            with open(densification_test_filename, 'wb') as f:
                pickle.dump(densification_infos_test, f)
            print('----------------Waymo densification info test file is saved to %s----------------' % densification_test_filename)
    print('---------------Data preparation Done---------------')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_waymo_infos', help='')
    parser.add_argument('--sampled_interval', type=int, default=1, help='Sample interval for train and validation')
    parser.add_argument('--sampled_interval_gt', type=int, default=10, help='Sample interval for ground truth database')
    parser.add_argument('--densified_pc', action='store_true', default=False, help='apply point cloud densification')
    parser.add_argument('--both_returns', action='store_true', default=False, help='Use both LiDAR Returns')
    args = parser.parse_args()

    if args.func == 'create_waymo_infos':
        import yaml
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_waymo_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'waymo',
            save_path=ROOT_DIR / 'data' / 'waymo',
            raw_data_tag='raw_data',
            processed_data_tag=dataset_cfg.PROCESSED_DATA_TAG,
            sampled_interval=args.sampled_interval,
            sampled_interval_gt=args.sampled_interval_gt,
            max_sweeps=dataset_cfg.get('MAX_SWEEPS', 0),
            densified_pc=args.densified_pc,
            both_returns=args.both_returns,
        )
    
    if args.func == 'create_waymo_test_set_infos':
        import yaml
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_waymo_test_set_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'waymo',
            save_path=ROOT_DIR / 'data' / 'waymo',
            raw_data_tag='raw_data_test_set',
            processed_data_tag=dataset_cfg.get('PROCESSED_DATA_TAG_TEST_SET', 'waymo_processed_data_test_set'),
            sampled_interval=args.sampled_interval,
            max_sweeps=dataset_cfg.get('MAX_SWEEPS', 0),
            densified_pc=args.densified_pc,
            both_returns=args.both_returns
        )

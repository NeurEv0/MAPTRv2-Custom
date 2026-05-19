"""Dataset for pre-annotating parking-scene no-go boundaries.

Loads the pkls produced by `projects/custom/data_preprocess/build_pkl.py`
and exposes records in the contract MapTRv2's pipeline/model/losses expect.

Design choices baked in (matching the converter):
  * World-axis-aligned lidar (ego) frame. `lidar2img` is precomputed (10, 4, 4)
    per sample so no yaw is needed.
  * 2D polylines, num_points = 40, BEV range +/-25 m.
  * 5 classes: car-no-go, other-no-go, column-no-go, wall-no-go, outside-no-go.
  * `segment` and `other` annotations are dropped; z is dropped.

Evaluation matches MapTRv2: per-class chamfer-distance AP at thresholds
{0.5, 1.0, 1.5} via `projects.mmdet3d_plugin.datasets.map_utils.mean_ap`.
"""

import json
import os
import os.path as osp
import pickle
import tempfile

import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.datasets import Custom3DDataset
from shapely.geometry import LineString

from projects.mmdet3d_plugin.datasets.nuscenes_map_dataset import (
    LiDARInstanceLines,
)


@DATASETS.register_module()
class SpaceLaneDataset(Custom3DDataset):
    """In-house pre-annotation dataset for parking/garage no-go boundaries."""

    CLASSES = (
        'car-no-go',
        'other-no-go',
        'column-no-go',
        'wall-no-go',
        'outside-no-go',
    )
    MAPCLASSES = CLASSES

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=False,
                 test_mode=False,
                 bev_size=(200, 200),
                 pc_range=(-25.0, -25.0, -5.0, 25.0, 25.0, 3.0),
                 fixed_ptsnum_per_line=40,
                 padding_value=-10000,
                 eval_use_same_gt_sample_num_flag=False,
                 map_ann_file=None,
                 **kwargs):
        # These fields must be set before super().__init__() because the base
        # class calls self.load_annotations() during __init__ and load may
        # touch them.
        self.pc_range = list(pc_range)
        self.bev_size = bev_size
        self.fixed_num = fixed_ptsnum_per_line
        self.padding_value = padding_value
        self.eval_use_same_gt_sample_num_flag = eval_use_same_gt_sample_num_flag
        self.map_ann_file = map_ann_file
        self.patch_size = (
            self.pc_range[4] - self.pc_range[1],
            self.pc_range[3] - self.pc_range[0],
        )
        self._pkl_metainfo = None

        super().__init__(
            data_root=data_root or '.',
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes if classes is not None else list(self.CLASSES),
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )
        # If the user did not pass an explicit data_root, fall back to whatever
        # the pkl recorded at conversion time.
        if not data_root and self._pkl_metainfo:
            recorded = self._pkl_metainfo.get('data_root')
            if recorded:
                self.data_root = recorded

        self.NUM_MAPCLASSES = len(self.MAPCLASSES)

    @classmethod
    def get_map_classes(cls, map_classes=None):
        if map_classes is None:
            return list(cls.MAPCLASSES)
        if isinstance(map_classes, (tuple, list)):
            return list(map_classes)
        raise ValueError(
            f'Unsupported type {type(map_classes)} for map_classes')

    def load_annotations(self, ann_file):
        with open(ann_file, 'rb') as f:
            payload = pickle.load(f)
        self._pkl_metainfo = payload.get('metainfo', {})
        return payload['samples']

    # -------- per-sample plumbing --------
    def get_data_info(self, index):
        info = self.data_infos[index]
        token = '{}/{}'.format(info['sample_name'], info['frame_id'])
        img_paths = [osp.join(self.data_root, p) for p in info['img_filenames']]
        lidar2img_arr = np.asarray(info['lidar2img'])  # (10, 4, 4)
        lidar2img = [lidar2img_arr[i] for i in range(lidar2img_arr.shape[0])]

        # Defensive `can_bus`/`scene_token` fields: the MapTR detector reads
        # both unconditionally (even with `use_can_bus=False`). Treat each
        # `sample_name` as a scene (trajectory) so frames inherit a shared
        # token; can_bus stays zero because we have no IMU and the encoder
        # transformer ignores it under our config.
        ego_xy = np.asarray(info['ego_world_xy'], dtype=np.float64)
        can_bus = np.zeros(18, dtype=np.float64)
        can_bus[:2] = ego_xy

        input_dict = dict(
            sample_idx=token,
            pts_filename=None,
            img_filename=img_paths,
            lidar2img=lidar2img,
            # composite lidar2img already encodes K; identity placeholder kept
            # for code paths that read cam_intrinsic.
            cam_intrinsic=[np.eye(4, dtype=np.float32) for _ in img_paths],
            ego_world_xy=ego_xy.astype(np.float32),
            sample_name=info['sample_name'],
            frame_id=info['frame_id'],
            image_type=info.get('image_type', {}),
            scene_token=info['sample_name'],
            can_bus=can_bus,
        )
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and len(annos['gt_vecs_label']) == 0:
                return None
        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        polylines = np.asarray(info['gt_polylines'])
        labels = np.asarray(info['gt_labels'], dtype=np.int64)
        line_strs = [LineString(p.tolist()) for p in polylines]
        gt_lines = LiDARInstanceLines(
            instance_line_list=line_strs,
            sample_dist=1,
            num_samples=250,
            padding=False,
            fixed_num=self.fixed_num,
            padding_value=self.padding_value,
            patch_size=self.patch_size,
        )
        return dict(gt_vecs_pts_loc=gt_lines, gt_vecs_label=labels)

    def vectormap_pipeline(self, example, input_dict):
        annos = input_dict['ann_info']
        example['gt_labels_3d'] = DC(
            to_tensor(annos['gt_vecs_label']), cpu_only=False)
        example['gt_bboxes_3d'] = DC(annos['gt_vecs_pts_loc'], cpu_only=True)
        return example

    @staticmethod
    def _wrap_queue_one(example):
        """Pack a single frame as a queue_length=1 batch element.

        MapTR/MapTRv2 expect img to be (queue, N_cam, C, H, W) per-sample and
        img_metas to be a dict keyed by queue index. With queue_length=1 we
        simply add the unit queue dim.
        """
        img_dc = example['img']
        img_tensor = img_dc.data  # (N_cam, C, H, W)
        example['img'] = DC(img_tensor.unsqueeze(0),
                            cpu_only=False, stack=True)
        meta = example['img_metas'].data
        example['img_metas'] = DC({0: meta}, cpu_only=True)
        return example

    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        example = self.vectormap_pipeline(example, input_dict)
        if self.filter_empty_gt and not (
                example['gt_labels_3d']._data != -1).any():
            return None
        return self._wrap_queue_one(example)

    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return self._wrap_queue_one(example)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    # -------- evaluation --------
    def _format_gt(self, gt_path):
        gt_annos = []
        for info in self.data_infos:
            token = '{}/{}'.format(info['sample_name'], info['frame_id'])
            polylines = np.asarray(info['gt_polylines'])
            labels = np.asarray(info['gt_labels'], dtype=np.int64)
            vectors = []
            for pts, lab in zip(polylines, labels):
                lab = int(lab)
                vectors.append(dict(
                    pts=pts.tolist(),
                    pts_num=int(len(pts)),
                    cls_name=self.CLASSES[lab],
                    type=lab,
                ))
            gt_annos.append(dict(sample_token=token, vectors=vectors))
        with open(gt_path, 'w') as f:
            json.dump({'GTs': gt_annos}, f)
        return gt_annos

    def _format_predictions(self, results, pred_path):
        pred_annos = []
        for i, det in enumerate(results):
            info = self.data_infos[i]
            token = '{}/{}'.format(info['sample_name'], info['frame_id'])
            d = det.get('pts_bbox', det)
            pts = d['pts_3d']
            scores = d['scores_3d']
            labels = d['labels_3d']
            if hasattr(pts, 'numpy'):
                pts = pts.numpy()
            if hasattr(scores, 'numpy'):
                scores = scores.numpy()
            if hasattr(labels, 'numpy'):
                labels = labels.numpy()
            pts = np.asarray(pts)
            scores = np.asarray(scores)
            labels = np.asarray(labels)
            vectors = []
            for j in range(len(scores)):
                lab = int(labels[j])
                vectors.append(dict(
                    pts=pts[j].tolist(),
                    pts_num=int(len(pts[j])),
                    cls_name=self.CLASSES[lab],
                    type=lab,
                    confidence_level=float(scores[j]),
                ))
            pred_annos.append(dict(sample_token=token, vectors=vectors))
        with open(pred_path, 'w') as f:
            json.dump({'meta': self.modality, 'results': pred_annos}, f)
        return pred_annos

    def evaluate(self,
                 results,
                 metric='chamfer',
                 logger=None,
                 jsonfile_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 **kwargs):
        from projects.mmdet3d_plugin.datasets.map_utils.mean_ap import (
            eval_map, format_res_gt_by_classes,
        )

        tmp_dir = None
        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = tmp_dir.name
        os.makedirs(jsonfile_prefix, exist_ok=True)

        pred_path = osp.join(jsonfile_prefix, 'space_pred.json')
        gt_path = osp.join(jsonfile_prefix, 'space_gt.json')
        pred_annos = self._format_predictions(results, pred_path)
        gt_annos = self._format_gt(gt_path)

        cls_gens, cls_gts = format_res_gt_by_classes(
            pred_path,
            pred_annos,
            gt_annos,
            cls_names=self.CLASSES,
            num_pred_pts_per_instance=self.fixed_num,
            eval_use_same_gt_sample_num_flag=self.eval_use_same_gt_sample_num_flag,
            pc_range=self.pc_range,
        )

        metrics_list = metric if isinstance(metric, list) else [metric]
        allowed = {'chamfer'}
        detail = {}
        for m in metrics_list:
            if m not in allowed:
                raise KeyError(
                    f'metric {m} not supported; allowed: {sorted(allowed)}')
            thresholds = [0.5, 1.0, 1.5]
            cls_aps = np.zeros((len(thresholds), self.NUM_MAPCLASSES))
            for ti, thr in enumerate(thresholds):
                _, cls_ap = eval_map(
                    pred_annos,
                    gt_annos,
                    cls_gens,
                    cls_gts,
                    threshold=thr,
                    cls_names=self.CLASSES,
                    logger=logger,
                    num_pred_pts_per_instance=self.fixed_num,
                    pc_range=self.pc_range,
                    metric=m,
                )
                for j in range(self.NUM_MAPCLASSES):
                    cls_aps[ti, j] = cls_ap[j]['ap']

            for i, name in enumerate(self.CLASSES):
                detail[f'SpaceLane_{m}/{name}_AP'] = float(cls_aps.mean(0)[i])
                for j, thr in enumerate(thresholds):
                    detail[f'SpaceLane_{m}/{name}_AP_thr_{thr}'] = float(
                        cls_aps[j, i])
            detail[f'SpaceLane_{m}/mAP'] = float(cls_aps.mean(0).mean())

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return detail

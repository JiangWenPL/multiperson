import os.path as osp

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         SegMapTransform, Numpy2Tensor)
from .utils import to_tensor, random_scale, flip_kp, flip_pose
from pycocotools.coco import COCO
from .extra_aug import ExtraAugmentation
from .custom import CustomDataset
import os.path as osp
import tensorboardX
import math
import json
import pickle
import matplotlib.pyplot as plt
from mmdetection.mmdet.models.utils.smpl.viz import draw_skeleton, J24_TO_J14
import random
import cv2
import torch
from .transforms import coco17_to_superset
from .h36m import H36MDataset

from .h36m import denormalize


class COCOKeypoints(H36MDataset):
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4),
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    # CLASSES = None
    CLASSES = ('Human',)

    def __init__(self,
                 filter_kpts=False,
                 **kwargs,
                 ):
        self.filter_kpts = filter_kpts
        super(COCOKeypoints, self).__init__(**kwargs)

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size and self.coco.getAnnIds(imgIds=[img_info['id']]):
                valid_inds.append(i)
        return valid_inds

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info, self.with_mask, self.with_kpts2d)

    def _parse_ann_info(self, ann_info, with_mask=True, with_kpts2d=True):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []
        if with_kpts2d:
            gt_kpts2d = list()
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
            if with_mask:
                gt_masks.append(self.coco.annToMask(ann))
                mask_polys = [
                    p for p in ann['segmentation'] if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)
            if with_kpts2d:
                np_kpts2d = np.array(ann['keypoints']).reshape(-1, 3)
                np_kpts2d[np_kpts2d[:, -1] > 0, -1] = 1
                if self.filter_kpts and np_kpts2d[..., -1].sum() < 8:
                    np_kpts2d = np.zeros_like(np_kpts2d)
                gt_kpts2d.append(np_kpts2d)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens

        if with_kpts2d:
            ann['kpts2d'] = coco17_to_superset(np.stack(gt_kpts2d)).astype(np.float32).copy()
            num_persons = ann['kpts2d'].shape[0]
            ann['kpts3d'] = np.zeros((num_persons, 24, 4), dtype=np.float32)
            ann['pose'] = np.zeros((num_persons, 24, 3), dtype=np.float32)
            ann['shape'] = np.zeros((num_persons, 10), dtype=np.float32)
            ann['trans'] = np.zeros((num_persons, 3), dtype=np.float32)
            ann['has_smpl'] = np.zeros((num_persons), dtype=np.float32)
        return ann

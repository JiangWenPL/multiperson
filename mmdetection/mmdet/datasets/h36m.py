import os.path as osp

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         SegMapTransform, Numpy2Tensor)
from .utils import to_tensor, random_scale, flip_kp, flip_pose, rot_aa
from .extra_aug import ExtraAugmentation
from .custom import CustomDataset
import os.path as osp
import tensorboardX
import math
import json
import pickle
import matplotlib.pyplot as plt
from mmdetection.mmdet.models.utils.smpl_utils import batch_rodrigues
import random
import cv2
from copy import deepcopy
import scipy
import scipy.misc
import seaborn as sns
import torch
import png

denormalize = lambda x: x * np.array([0.229, 0.224, 0.225])[None, None, :] + np.array([0.485, 0.456, 0.406])[None, None,
                                                                             :]
import pycocotools.mask as mask_util


def rot2DPts(x, y, rotMat):
    new_x = rotMat[0, 0] * x + rotMat[0, 1] * y + rotMat[0, 2]
    new_y = rotMat[1, 0] * x + rotMat[1, 1] * y + rotMat[1, 2]
    return new_x, new_y


class H36MDataset(Dataset):
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
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 multiscale_mode='value',
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 with_semantic_seg=False,
                 seg_prefix=None,
                 seg_scale_factor=1,
                 extra_aug=None,
                 resize_keep_ratio=True,
                 test_mode=False,
                 with_shape=True,
                 with_kpts3d=True,
                 with_kpts2d=True,
                 with_pose=True,
                 with_trans=True,
                 max_samples=-1,  # Commonly used in validating
                 noise_factor=0.4,
                 square_bbox=True,
                 rot_factor=0,
                 sample_weight=1,
                 with_dp=False,
                 mosh_path=None,
                 ignore_3d=False,
                 ignore_smpl=False,
                 with_nr=False,
                 sample_by_persons=False,
                 use_poly=False,
                 **kwargs,
                 ):
        # prefix of images path
        self.img_prefix = img_prefix

        # load annotations (and proposals)
        self.img_infos = self.load_annotations(ann_file)
        self.max_samples = max_samples

        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None
        # filter images with no annotation during training
        if not test_mode and False:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]

        # Select a subset for quick validation
        if self.max_samples > 0:
            self.img_infos = random.sample(self.img_infos, max_samples)
            # self.img_infos = self.img_infos[:max_samples]

        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = img_norm_cfg

        # multi-scale mode (only applicable for multi-scale training)
        self.multiscale_mode = multiscale_mode
        assert multiscale_mode in ['value', 'range']

        # max proposals per image
        self.num_max_proposals = num_max_proposals
        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor

        # with mask or not (reserved field, takes no effect)
        self.with_mask = with_mask
        # some datasets provide bbox annotations as ignore/crowd/difficult,
        # if `with_crowd` is True, then these info is returned.
        self.with_crowd = with_crowd
        # with label is False for RPN
        self.with_label = with_label
        # with semantic segmentation (stuff) annotation or not
        self.with_seg = with_semantic_seg
        # prefix of semantic segmentation map path
        self.seg_prefix = seg_prefix
        # rescale factor for segmentation maps
        self.seg_scale_factor = seg_scale_factor
        # in test mode or not
        self.test_mode = test_mode
        # For dataset with SMPL shape parameters
        self.with_shape = with_shape
        # For dataset with joints rotation matrix in SMPL model
        self.with_kpts3d = with_kpts3d
        # For dataset with 2D pose
        self.with_kpts2d = with_kpts2d
        # For dataset with camera parameters in SMPL model
        self.with_trans = with_trans
        # For pose in axis angle of the joints
        self.with_pose = with_pose

        # Densepose annotations
        self.with_dp = with_dp

        # noise factor for color jittering
        self.noise_factor = noise_factor

        # Whether to adjust bbox to square manually..
        self.square_bbox = square_bbox

        # Rotation facotr
        self.rot_factor = rot_factor

        # Mosh dataset for generator
        self.mosh_path = None  # mosh_path

        # set group flag for the sampler
        if not self.test_mode and False:
            self._set_group_flag()
        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.bbox_transform = BboxTransform()
        self.mask_transform = MaskTransform()
        self.seg_transform = SegMapTransform(self.size_divisor)
        self.numpy2tensor = Numpy2Tensor()

        # if use extra augmentation
        if extra_aug is not None:
            self.extra_aug = ExtraAugmentation(**extra_aug)
        else:
            self.extra_aug = None

        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio

        self.sample_weight = sample_weight

        if sample_by_persons:
            persons_cnt = np.zeros(len(self.img_infos))
            for i in range(len(self.img_infos)):
                persons_cnt[i] = self.get_ann_info(i)['kpts2d'].shape[0]
            self.density = sample_weight * persons_cnt / persons_cnt.sum()
        else:
            self.density = sample_weight * np.ones(len(self.img_infos)) / len(self.img_infos)

        self.ignore_3d = ignore_3d
        self.ignore_smpl = ignore_smpl
        self.with_nr = with_nr
        self.use_poly = use_poly

        if self.mosh_path:
            mosh = np.load(mosh_path)
            self.mosh_shape = mosh['shape'].copy()
            self.mosh_pose = mosh['pose'].copy()
            self.mosh_sample_list = range(self.mosh_shape.shape[0])

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file):
        """
        filename:
        height: 1000
        width: commonly 1002 in h36m
        :param ann_file:
        :return:
        """
        with open(ann_file, 'rb') as f:
            raw_infos = pickle.load(f)
        return raw_infos

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        """
        :param idx:
        :return:A dict of the following iterms:
            bboxes: [x1, y1, x2, y2]
            labels: number
            kpts3d: (24, 4)
            kpts2d: (24, 3)
            pose: (72,)
            shape: (10,)
            cam: (3,) (The trans in SMPL model)
        """
        # Visualization needed
        raw_info = deepcopy(self.img_infos[idx])
        bbox = raw_info['bbox']
        if self.square_bbox:
            center = np.array([int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)])
            bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            half_size = int(math.floor(bbox_size / 2))
            square_bbox = np.array(
                [center[0] - half_size, center[1] - half_size, center[0] + half_size, center[1] + half_size])
        else:
            square_bbox = np.array(bbox)
        #  Here comes a problem, what if the bbox overflows on the corner?
        # They will be args passed to `model.train_forward` if we overwrite `CustomDataset`.
        return {'bboxes': square_bbox.reshape(1, -1).astype(np.float32),  # (1,4)
                # There will be only one person here.
                'labels': np.array([1]),
                'kpts3d': raw_info['S'][np.newaxis].astype(np.float32),  # (1, 24,4) extra chanel for visibility
                'kpts2d': raw_info['part'][np.newaxis].astype(np.float32),  # (1, 24,3) extra chanel for visibility
                'pose': raw_info['pose'].reshape(-1, 3)[np.newaxis].astype(np.float32),  # (1, 24, 3)
                'shape': raw_info['shape'][np.newaxis].astype(np.float32),  # (1,10)
                'trans': raw_info['trans'][np.newaxis].astype(np.float32),  # (1, 3)
                'has_smpl': np.array([1])
                }  # I think `1` represents the first class

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        # if self.test_mode:
        #     return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    @staticmethod
    def val(runner, dataloader, **kwargs):
        from IPython import embed
        embed()
        pass

    @staticmethod
    def annToRLE(ann):
        h, w = ann['height'], ann['width']
        segm = ann['segmentation']
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = mask_util.frPyObjects(segm, h, w)
            rle = mask_util.merge(rles)
        elif type(segm['counts']) == list:
            # uncompressed RLE
            rle = mask_util.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def prepare_train_img(self, idx):
        img_info = deepcopy(self.img_infos[idx])
        # load image
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        img_info['height'], img_info['width'] = img.shape[:2]
        raw_shape = img.shape
        # Color jittering:
        pn = np.random.uniform(1 - self.noise_factor, 1 + self.noise_factor, 3)
        img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, img[:, :, 0] * pn[0]))
        img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, img[:, :, 1] * pn[1]))
        img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, img[:, :, 2] * pn[2]))

        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        ann = self.get_ann_info(idx)
        if self.ignore_3d:
            ann['kpts3d'] = np.zeros_like(ann['kpts3d'])

        if self.ignore_smpl:
            ann['has_smpl'] = np.zeros_like(ann['has_smpl'])

        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # extra augmentation
        if self.extra_aug is not None:
            img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes,
                                                       gt_labels)

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)

        img, img_shape, pad_shape, scale_factor = self.img_transform(img, img_scale, flip,
                                                                     keep_ratio=self.resize_keep_ratio)

        # Force padding for the issue of multi-GPU training
        padded_img = np.zeros((img.shape[0], img_scale[1], img_scale[0]), dtype=img.dtype)
        padded_img[:, :img.shape[-2], :img.shape[-1]] = img
        img = padded_img

        if self.with_nr:
            padded_scene = np.zeros((img.shape[-2], img.shape[-1]), dtype=np.uint8)

        if self.with_nr:
            has_mask = True
            if 'dp_mask_path' in ann:
                raw_mask = cv2.imread(osp.join(self.img_prefix, ann['dp_mask_path']), cv2.IMREAD_UNCHANGED)
            elif 'segmentation' in ann and self.use_poly:
                assert 'COCO' in ann['filename'], "Only support coco segmentation now"
                raw_mask = np.zeros((ann['height'], ann['width']), dtype=np.uint8)
                for i, seg in enumerate(ann['segmentation']):
                    ori_mask = mask_util.decode(
                        self.annToRLE({'width': ann['width'], 'height': ann['height'], 'segmentation': seg}))
                    raw_mask[ori_mask > 0] = i + 1
            else:
                has_mask = False
            padded_scene = np.zeros((img.shape[-2], img.shape[-1]), dtype=np.uint8)
            if has_mask:
                target_shape = int(np.round(raw_shape[1] * scale_factor)), int(np.round(raw_shape[0] * scale_factor))
                resized_mask = cv2.resize(raw_mask, target_shape, interpolation=cv2.INTER_NEAREST)
                if flip:
                    resized_mask = np.flip(resized_mask, axis=1)
                padded_scene[:resized_mask.shape[-2], :resized_mask.shape[-1]] = resized_mask

        if self.with_seg:
            gt_seg = mmcv.imread(
                osp.join(self.seg_prefix, img_info['file_name'].replace(
                    'jpg', 'png')),
                flag='unchanged')
            gt_seg = self.seg_transform(gt_seg.squeeze(), img_scale, flip)
            gt_seg = mmcv.imrescale(
                gt_seg, self.seg_scale_factor, interpolation='nearest')
            gt_seg = gt_seg[None, ...]
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack(
                [proposals, scores]) if scores is not None else proposals
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            gt_masks = self.mask_transform(ann['masks'], pad_shape,
                                           scale_factor, flip)

        if self.with_shape:
            gt_shapes = ann['shape']
        if self.with_kpts2d:
            gt_kpts2d = ann['kpts2d']
            # Rescale the 2D keypoints now.
            s_kpts2d = np.zeros_like(gt_kpts2d)
            s_kpts2d[..., -1] = gt_kpts2d[..., -1]
            s_kpts2d[..., :-1] = gt_kpts2d[..., :-1] * scale_factor
            gt_kpts2d = s_kpts2d

            if flip:
                for i, kp in enumerate(gt_kpts2d):
                    gt_kpts2d[i] = flip_kp(kp, img_shape[1])  # img is (C, H, W)
                    # NOTE: I use the img_shape to avoid the influence of padding.

        if self.with_kpts3d:
            gt_kpts3d = ann['kpts3d']
            if flip:
                for i, kp in enumerate(gt_kpts3d):
                    gt_kpts3d[i] = flip_kp(kp, 0)  # Not the image width as the pose is centered by hip.
        if self.with_trans:
            gt_trans = ann['trans']
        if self.with_pose:
            gt_poses = ann['pose']
            if flip:
                for i, ps in enumerate(gt_poses):
                    gt_poses[i] = flip_pose(ps.reshape(-1)).reshape(-1, 3)

        if self.with_dp:
            dp_num_pts = ann['dp_num_pts']
            dp_x = ann['dp_x']
            dp_y = ann['dp_y']
            dp_U = ann['dp_U']
            dp_V = ann['dp_V']
            dp_I = ann['dp_I']
            dp_x = img_shape[1] - dp_x

        if not self.rot_factor == 0 and np.random.uniform() > 0.6:
            rot = min(2 * self.rot_factor,
                      max(-2 * self.rot_factor, np.random.randn() * self.rot_factor))
            rot_rad = -rot * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat = np.eye(3)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]

            # As we are rotating around the center.
            t_mat = np.eye(3)
            res = img.shape[1:]
            t_mat[0, 2] = -res[1] / 2
            t_mat[1, 2] = -res[0] / 2
            t_inv = t_mat.copy()
            t_inv[:2, 2] *= -1
            rot_mat2d = t_inv @ rot_mat @ t_mat
            # Things to change:
            # kpts2d and 3d
            # img and bbox

            # On GraphCMR, they only rotate on the first 3 entries. So I think it apply to ours.
            for i in range(gt_poses.shape[0]):
                gt_poses[i, 0] = rot_aa(gt_poses[i, 0], rot)
                gt_kpts3d[i, :, :-1] = (rot_mat @ gt_kpts3d[i, :, :-1].T).T
                gt_kpts2d[i, :, :-1] = (rot_mat2d @ gt_kpts2d[i, :, ].T).T[..., :-1]
                if sum(gt_kpts2d[i, ..., -1]) > 0:
                    x_min, y_min, _ = gt_kpts2d[i, gt_kpts2d[i, ..., -1] > 0].min(0)
                    x_max, y_max, _ = gt_kpts2d[i, gt_kpts2d[i, ..., -1] > 0].max(0)
                else:
                    # If there is no valida ktps, we will use gt_bboxes instead.
                    x_min, y_min = rot2DPts(gt_bboxes[i][0], gt_bboxes[i][1], rot_mat2d)
                    x_max, y_max = rot2DPts(gt_bboxes[i][2], gt_bboxes[i][3], rot_mat2d)
                bbox_w, bbox_h = x_max - x_min, y_max - y_min
                bbox_center = (x_min + x_max) / 2, (y_min + y_max) / 2
                _, im_w, im_h = img.shape
                x_min, y_min = max(0, bbox_center[0] - bbox_w * 0.55), max(0, bbox_center[1] - bbox_h * 0.55)
                x_max, y_max = min(im_w, bbox_center[0] + bbox_w * 0.55), min(im_h, bbox_center[1] + bbox_h * 0.55)
                bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
                if self.square_bbox:
                    center = np.array([int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)])
                    bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
                    half_size = int(math.floor(bbox_size / 2))
                    square_bbox = np.array(
                        [center[0] - half_size, center[1] - half_size, center[0] + half_size, center[1] + half_size])
                    bbox = square_bbox.astype(np.float32).copy()
                gt_bboxes[i] = bbox
                if self.with_dp:
                    dp_x, dp_y = rot2DPts(dp_x, dp_y, rot_mat2d)

            img_raw = cv2.cvtColor((denormalize(img.transpose([1, 2, 0])) * 255).astype(np.uint8).copy(),
                                   cv2.COLOR_BGR2RGB)
            img_rotated = scipy.misc.imrotate(img_raw, rot)

            # For debug

            img = mmcv.imnormalize(img_rotated, self.img_transform.mean, self.img_transform.std,
                                   self.img_transform.to_rgb).transpose([2, 0, 1])
        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip,
            idx=idx,
            file_name=img_info['filename']
        )
        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)))
        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)
        if self.with_seg:
            data['gt_semantic_seg'] = DC(to_tensor(gt_seg), stack=True)
        if self.with_pose:
            data['gt_poses'] = DC(to_tensor(gt_poses))
        if self.with_kpts2d:
            # Filter out un-used kpts.
            kpts_filter = gt_kpts2d[..., -1].sum(-1) < 8
            if gt_kpts2d.shape[0] == 1:
                gt_kpts2d[kpts_filter] = 0
            data['gt_kpts2d'] = DC(to_tensor(gt_kpts2d))
        if self.with_kpts3d:
            data['gt_kpts3d'] = DC(to_tensor(gt_kpts3d))
        if self.with_shape:
            data['gt_shapes'] = DC(to_tensor(gt_shapes))
        if self.with_trans:
            data['gt_trans'] = DC(to_tensor(gt_trans))
        if self.with_dp:
            data['dp_x'] = DC(to_tensor(dp_x))
            data['dp_y'] = DC(to_tensor(dp_y))
            data['dp_U'] = DC(to_tensor(dp_U))
            data['dp_V'] = DC(to_tensor(dp_V))
            data['dp_I'] = DC(to_tensor(dp_I))
            data['dp_num_pts'] = DC(to_tensor(dp_num_pts))
        # if self.with_shape = DC(to_tensor(gt_shape))
        if self.with_nr:
            data['scene'] = DC(to_tensor(padded_scene))
        data['has_smpl'] = DC(to_tensor(ann['has_smpl']))
        if self.mosh_path:
            sampled_idxs = np.array(random.sample(self.mosh_sample_list, 36))
            mosh_pose = torch.tensor(deepcopy(self.mosh_pose[sampled_idxs].astype(np.float32)))
            mosh_shape = torch.tensor(deepcopy(self.mosh_shape[sampled_idxs].astype(np.float32)))
            mosh_bs = mosh_shape.shape[0]
            mosh_pose_shape = torch.cat([batch_rodrigues(mosh_pose.view(-1, 3)).view(mosh_bs, -1), mosh_shape], dim=1)
            data['mosh'] = DC(mosh_pose_shape)
        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        if self.proposals is not None:
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None

        def prepare_single(img, scale, flip, proposal=None):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img, scale, False, proposal)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs, img_meta=img_metas)
        if self.proposals is not None:
            data['proposals'] = proposals
        return data

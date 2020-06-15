import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin, SMPLTestMixin
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler, delta2bbox
from mmdet.core import tensor2imgs, get_classes, auto_fp16
import numpy as np


@DETECTORS.register_module
class SMPLRCNN(BaseDetector, RPNTestMixin, BBoxTestMixin,
               MaskTestMixin, SMPLTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 smpl_roi_extractor=None,
                 smpl_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 smpl_weight=1,
                 other_weight=1.,
                 gt_pos_only=False,
                 debugging=False,
                 hard_square=False,
                 self_mask_weight=1.,
                 inter_mask_weight=1.,
                 kpts_head=None,
                 kpts_roi_extractor=None,
                 ):
        super(SMPLRCNN, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        if smpl_head is not None:
            if smpl_roi_extractor is not None:
                self.smpl_roi_extractor = builder.build_roi_extractor(
                    smpl_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.smpl_roi_extractor = self.bbox_roi_extractor
            self.smpl_head = builder.build_head(smpl_head)
            self.smpl_weight = smpl_weight
            self.other_weight = other_weight
            self.gt_pos_only = gt_pos_only
            self.bbox_feat = hasattr(self.smpl_head, 'bbox_feat') and self.smpl_head.bbox_feat

        if kpts_head is not None:
            if kpts_roi_extractor is not None:
                self.kpts_roi_extractor = builder.build_roi_extractor(
                    kpts_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.kpts_roi_extractor = self.bbox_roi_extractor
            self.kpts_head = builder.build_head(kpts_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.debugging = debugging
        self.hard_square = hard_square
        self.self_mask_weight = self_mask_weight
        self.inter_mask_weight = inter_mask_weight

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_smpl(self):
        return hasattr(self, 'smpl_head') and self.smpl_head is not None

    @property
    def with_kpts(self):
        return hasattr(self, 'kpts_head') and self.smpl_head is not None

    def init_weights(self, pretrained=None):
        super(SMPLRCNN, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_smpl:
            self.smpl_head.init_weights()
            if not self.share_roi_extractor:
                self.smpl_roi_extractor.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_kpts3d=None,
                      gt_kpts2d=None,
                      gt_shapes=None,
                      gt_poses=None,
                      gt_trans=None,
                      has_smpl=None,
                      proposals=None,
                      disable_smpl=False,
                      return_pred=False,
                      dp_x=None,
                      dp_y=None,
                      dp_U=None,
                      dp_V=None,
                      dp_I=None,
                      dp_num_pts=None,
                      scene=None,
                      log_depth=None,
                      **kwargs):
        if self.debugging:
            import ipdb
            ipdb.set_trace()
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

        # Just for debugging
        if sum(rpn_losses['loss_rpn_bbox']) > 10:
            print(rpn_losses)
            print(gt_bboxes)
            print(loss_bbox)

        # mask head forward and loss
        if self.with_smpl:
            if not self.share_roi_extractor:
                if not self.gt_pos_only:
                    pos_rois = bbox2roi(
                        [res.pos_bboxes for res in sampling_results])
                else:
                    pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
                if self.hard_square:
                    corner_pts = pos_rois[:, 1:].clone()
                    center_pts = 0.5 * (corner_pts[:, :2] + corner_pts[:, 2:])
                    rois_size, _ = torch.max(corner_pts[:, 2:] - corner_pts[:, :2], dim=1)
                    rois_size *= 0.5
                    new_corners = torch.stack(
                        [center_pts[:, 0] - rois_size, center_pts[:, 1] - rois_size, center_pts[:, 0] + rois_size,
                         center_pts[:, 1] + rois_size], dim=-1)
                    new_corners = torch.clamp(new_corners, 0, max(img.shape[-2:]))
                    pos_rois = pos_rois.detach().clone()
                    pos_rois[:, 1:] = new_corners
                mask_feats = self.smpl_roi_extractor(
                    x[:self.smpl_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]

            # To get the confidence from detection head.
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            bboxes_confidence = cls_score[pos_inds, 1]
            pred_bboxes, kpts2d_target, kpts3d_target, poses_target, shapes_target, trans_target, has_smpl_target, gt_vertices, idxs_in_batch, pose_idx = self.smpl_head.get_target(
                sampling_results, gt_kpts2d, gt_kpts3d, gt_poses, gt_shapes, gt_trans, has_smpl,
                self.train_cfg.rcnn)
            smpl_targets = {
                'gt_keypoints_2d': kpts2d_target,
                'gt_keypoints_3d': kpts3d_target,
                'gt_rotmat': poses_target,
                'gt_shape': shapes_target,
                'gt_camera': trans_target,
                'has_smpl': has_smpl_target,
                'gt_vertices': gt_vertices,
                'pred_bboxes': pred_bboxes,
                'raw_images': img.clone(),
                'img_meta': img_meta,
                'idxs_in_batch': idxs_in_batch,
                'pose_idx': pose_idx,
                'mosh': kwargs.get('mosh', None),
                'scene': scene,
                'log_depth': log_depth,
            }
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            if hasattr(self.smpl_head, 'bbox_feat') and self.smpl_head.bbox_feat:
                im_w, im_h = img.shape[-2:]
                bboxes_w = torch.clamp(pred_bboxes[..., 2] - pred_bboxes[..., 0], 1, im_w)
                bboxes_h = torch.clamp(pred_bboxes[..., 3] - pred_bboxes[..., 1], 1, im_h)
                bboxes_aspect_ratio = bboxes_w / bboxes_h
                bbox_info_feat = torch.stack([bboxes_w, bboxes_h, bboxes_aspect_ratio], dim=1)
                smpl_pred = self.smpl_head((mask_feats, bbox_info_feat))
            else:
                smpl_pred = self.smpl_head(mask_feats)

            loss_smpl = self.smpl_head.loss(smpl_pred, smpl_targets,
                                            pos_labels, bboxes_confidence=bboxes_confidence,
                                            discriminator=kwargs.get('discriminator', None))
            if self.other_weight != 1.:
                for k in losses:
                    if 'loss' in k:
                        losses[k] *= self.other_weight

            # Mute the loss of smpl head
            if self.smpl_weight != 1:
                for k in loss_smpl:
                    if 'loss' in k:
                        loss_smpl[k] *= self.smpl_weight
            losses.update(loss_smpl)

            # Fox bboxes from detection head
            pred_delta = bbox_pred[pos_inds, 4:8].detach()
            bbox_out = delta2bbox(pred_bboxes, pred_delta, self.bbox_head.target_means, self.bbox_head.target_stds)

            gt_delta = bbox_targets[2][pos_inds].detach()
            gt_out = delta2bbox(pred_bboxes, gt_delta, self.bbox_head.target_means, self.bbox_head.target_stds)
            losses.update({'img$head_bboxes': bbox_out,
                           'img$gt_bboxes': gt_out,
                           })

        if self.with_kpts:
            kpts_pred = self.kpts_head(mask_feats)

            kpts_target = self.kpts_head.get_target(
                sampling_results, gt_kpts2d, self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_kpts = self.kpts_head.loss(kpts_pred, kpts_target,
                                            pos_labels)
            # np_heatmap = kpts_pred.detach().cpu().numpy()
            # x, y = np.where((np_heatmap.max((-2, -1))[..., None, None] == np_heatmap))
            losses.update(loss_kpts)

        if return_pred:
            return losses, smpl_pred
        else:
            return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False, use_gt_bboxes=False, gt_bboxes=None, **kwargs):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)

        if use_gt_bboxes:
            det_bboxes = gt_bboxes[0]  # 1 image/it for evaluation during training
            bbox_results = bbox2result(det_bboxes, torch.zeros(det_bboxes.shape[0]).to(det_bboxes.device),
                                       self.bbox_head.num_classes)
        else:
            proposal_list = self.simple_test_rpn(
                x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

            det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
            bbox_results = bbox2result(det_bboxes, det_labels,
                                       self.bbox_head.num_classes)
        # TODO: Add simple SMPL test here. Try to reuse function in SMPL loss.

        if not self.with_smpl:
            return bbox_results
        if self.with_smpl:
            smpl_results = self.simple_test_smpl(x, img_meta, det_bboxes, img.shape, rescale=rescale)
            return bbox_results, smpl_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results

    def forward_test(self, imgs, img_metas, **kwargs):

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            img_metas = [img_metas] # Hack for demo code
            #raise ValueError(
            #    'num of augmentations ({}) != num of image meta ({})'.format(
            #        len(imgs), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs.size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(imgs, img_metas, **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

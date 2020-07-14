import torch
import torch.nn as nn

from .utils import weighted_loss
from ..registry import LOSSES
from ..utils.smpl_utils import batch_rodrigues, perspective_projection
from ..utils.pose_utils import reconstruction_error
from ..utils.smpl.smpl import SMPL
import random
from sdf import SDFLoss
import neural_renderer as nr
import numpy as np


def batch_adv_disc_l2_loss(real_disc_value, fake_disc_value):
    ka = real_disc_value.shape[0]
    kb = fake_disc_value.shape[0]
    lb, la = torch.sum(fake_disc_value ** 2) / kb, torch.sum((real_disc_value - 1) ** 2) / ka
    return la, lb, la + lb


def batch_encoder_disc_l2_loss(disc_value):
    k = disc_value.shape[0]
    return torch.sum((disc_value - 1.0) ** 2) * 1.0 / k


@weighted_loss
def smpl_loss(pred, target, beta=1.0):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


@torch.no_grad()
def select_index(im_id, pids, metric, invalid_mask=None):
    im_id = im_id.clone().int()[:, 0]
    num_imgs = im_id.max().item()
    selected_idxs = list()
    full_idxs = torch.arange(im_id.shape[0], device=im_id.device)
    for bid in set(im_id.tolist()):
        batch_mask = bid == im_id
        cur_pids = pids[batch_mask]
        cur_select = list()
        for pid in set(cur_pids.tolist()):
            person_mask = (pid == cur_pids)
            idx_to_select = full_idxs[batch_mask][person_mask][metric[batch_mask][person_mask].argmax()]
            if invalid_mask and invalid_mask[idx_to_select]:
                continue
            cur_select.append(idx_to_select)
        selected_idxs.append(cur_select)
    return selected_idxs


def adversarial_loss(discriminator, pred_pose_shape, real_pose_shape):
    loss_disc = batch_encoder_disc_l2_loss(discriminator(pred_pose_shape))
    fake_pose_shape = pred_pose_shape.detach()
    fake_disc_value, real_disc_value = discriminator(fake_pose_shape), discriminator(real_pose_shape)
    d_disc_real, d_disc_fake, d_disc_loss = batch_adv_disc_l2_loss(real_disc_value, fake_disc_value)
    return loss_disc, d_disc_fake, d_disc_real


@LOSSES.register_module
class SMPLLoss(nn.Module):

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0, re_weight=None,
                 normalize_kpts=False, pad_size=False, JOINT_REGRESSOR_H36M='data/J_regressor_h36m.npy',
                 debugging=False, adversarial_cfg=None, use_sdf=False, FOCAL_LENGTH=1000,
                 kpts_loss_type='L1Loss', kpts_3d_loss_type=None, img_size=None,
                 nr_batch_rank=False, inner_robust_sdf=None, **kwargs):
        super(SMPLLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.criterion_shape = nn.L1Loss()
        self.criterion_keypoints = getattr(nn, kpts_loss_type)(reduction='none')  # nn.L1Loss(reduction='none')
        if kpts_3d_loss_type is not None:
            self.criterion_3d_keypoints = getattr(nn, kpts_3d_loss_type)(reduction='none')
        self.criterion_regr = nn.MSELoss()
        self.re_weight = re_weight
        self.normalize_kpts = normalize_kpts
        self.pad_size = pad_size
        self.FOCAL_LENGTH = FOCAL_LENGTH
        self.debugging = debugging
        # Initialize SMPL model
        self.smpl = SMPL('data/smpl')
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()

        self.adversarial_cfg = adversarial_cfg
        self.use_sdf = use_sdf
        if debugging:
            self.sdf_loss = SDFLoss(self.smpl.faces, debugging=self.debugging, robustifier=inner_robust_sdf)
        else:
            self.sdf_loss = SDFLoss(self.smpl.faces, robustifier=inner_robust_sdf)
        self.nr_batch_rank = nr_batch_rank
        if self.nr_batch_rank:
            # setup renderer
            self.image_size = max(img_size)
            self.w_diff, self.h_diff = (self.image_size - img_size[0]) // 2, (self.image_size - img_size[1]) // 2

            self.neural_renderer = nr.Renderer(dist_coeffs=None, orig_size=self.image_size,
                                               image_size=self.image_size,
                                               light_intensity_ambient=1,
                                               light_intensity_directional=0,
                                               anti_aliasing=False)

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                bboxes_confidence=None,
                discriminator=None,
                **kwargs):
        """

        :param pred: SMPL parameters with 24*6+10+3
        :param target: same as pred
        :param weight:
        :param avg_factor:
        :param reduction_override:
        :param kwargs:
        :param bboxes_confidence:
        :return: loss: dict. All the value whose keys contain 'loss' will be summed up.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        pred_rotmat = pred['pred_rotmat']
        pred_camera = pred['pred_camera']
        pred_joints = pred['pred_joints']
        pred_vertices = pred['pred_vertices']
        pred_betas = pred['pred_betas']

        gt_rotmat = target['gt_rotmat']  # It's not rotmat actually. This is a (B, 24, 3) tensor.
        gt_shape = target['gt_shape']
        gt_camera = target['gt_camera']
        gt_keypoints_2d = target['gt_keypoints_2d']
        gt_keypoints_3d = target['gt_keypoints_3d']
        has_smpl = target['has_smpl']
        gt_vertices = target['gt_vertices']
        pred_bboxes = target['pred_bboxes']
        raw_images = target['raw_images']
        img_meta = target['img_meta']
        ori_shape = [i['ori_shape'] for i in img_meta]
        idxs_in_batch = target['idxs_in_batch']
        pose_idx = target['pose_idx']
        scene = target['scene']
        batch_size = pred_joints.shape[0]
        if self.pad_size:
            img_pad_shape = torch.tensor([i['pad_shape'][:2] for i in img_meta], dtype=torch.float32).to(
                pred_joints.device)
            img_size = img_pad_shape[idxs_in_batch[:, 0].long()]
        else:
            img_size = torch.zeros(batch_size, 2).to(pred_joints.device)
            img_size += torch.tensor(raw_images.shape[:-3:-1], dtype=img_size.dtype).to(img_size.device)
        center_pts = (pred_bboxes[..., :2] + pred_bboxes[..., 2:]) / 2
        crop_translation = torch.zeros((batch_size, 3), dtype=pred_camera.dtype).to(
            pred_joints.device)
        crop_translation[..., :2] = pred_camera[..., 1:]
        # We may detach it.
        bboxes_size = torch.max(torch.abs(pred_bboxes[..., 0] - pred_bboxes[..., 2]),
                                torch.abs(pred_bboxes[..., 1] - pred_bboxes[..., 3]))
        valid_boxes = (torch.abs(pred_bboxes[..., 0] - pred_bboxes[..., 2]) > 5) & (torch.abs(
            pred_bboxes[..., 1] - pred_bboxes[..., 3]) > 5)
        crop_translation[..., 2] = 2 * self.FOCAL_LENGTH / (1e-6 + pred_camera[..., 0] * bboxes_size)
        rotation_Is = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(pred_joints.device)
        depth = 2 * self.FOCAL_LENGTH / (1e-6 + pred_camera[..., 0] * bboxes_size)
        translation = torch.zeros((batch_size, 3), dtype=pred_camera.dtype).to(
            pred_joints.device)
        translation[:, :-1] = depth[:, None] * (center_pts + pred_camera[:, 1:] * bboxes_size.unsqueeze(
            -1) - img_size / 2) / self.FOCAL_LENGTH
        translation[:, -1] = depth
        focal_length = self.FOCAL_LENGTH * torch.ones_like(depth)
        # pred_joints_translated = pred_joints + translation[:, None, :]
        pred_keypoints_2d_smpl = perspective_projection(pred_joints,
                                             rotation_Is,
                                             translation,
                                             focal_length,
                                             img_size / 2)
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        pred_keypoints_2d_smpl_orig = pred_keypoints_2d_smpl.clone()
        if self.normalize_kpts:
            scale_w = torch.clamp(pred_bboxes[..., 2] - pred_bboxes[..., 0], 1, img_size[..., 0].max())
            scale_h = torch.clamp(pred_bboxes[..., 3] - pred_bboxes[..., 1], 1, img_size[..., 1].max())
            bboxes_scale = torch.stack([scale_w, scale_h], dim=1)
            gt_keypoints_2d[..., :2] = (gt_keypoints_2d[..., :2] - center_pts.unsqueeze(1)) / bboxes_scale.unsqueeze(1)
            pred_keypoints_2d_smpl = (pred_keypoints_2d_smpl - center_pts.unsqueeze(1)) / bboxes_scale.unsqueeze(1)
        else:
            pred_keypoints_2d_smpl = pred_keypoints_2d_smpl / img_size.unsqueeze(1)
            gt_keypoints_2d[:, :, :-1] = gt_keypoints_2d[:, :, :-1] / img_size.unsqueeze(1)
        loss_keypoints_smpl, error_ranks = self.keypoint_loss(pred_keypoints_2d_smpl[valid_boxes],
                                                              gt_keypoints_2d[valid_boxes])

        loss_keypoints_3d_smpl = self.keypoint_3d_loss(pred_joints[valid_boxes], gt_keypoints_3d[valid_boxes])
        loss_shape_smpl = self.shape_loss(pred_vertices[valid_boxes], gt_vertices[valid_boxes], has_smpl[valid_boxes])
        loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat[valid_boxes], pred_betas[valid_boxes],
                                                           gt_rotmat[valid_boxes], gt_shape[valid_boxes],
                                                           has_smpl[valid_boxes])

        loss_dict = {'loss_keypoints_smpl': loss_keypoints_smpl * 4, 'loss_keypoints_3d_smpl': loss_keypoints_3d_smpl,
                     'loss_shape_smpl': loss_shape_smpl, 'loss_regr_pose': loss_regr_pose,
                     'loss_regr_betas': loss_regr_betas * 0.01,
                     'img$raw_images': raw_images.detach(), 'img$idxs_in_batch': idxs_in_batch.detach(),
                     'img$pose_idx': pose_idx.detach(),
                     'img$pred_vertices': pred_vertices.detach(),
                     'img$translation': translation.detach(), 'img$error_rank': -bboxes_confidence.detach(),
                     'img$pred_bboxes': pred_bboxes.detach(),
                     'img$pred_keypoints_2d_smpl': (pred_keypoints_2d_smpl_orig[:, -24:, :]).detach().clone(),
                     'img$gt_keypoints_2d': gt_keypoints_2d_orig.detach().clone(),
                     }

        if self.adversarial_cfg:
            valid_batch_size = pred_rotmat[valid_boxes].shape[0]
            pred_pose_shape = torch.cat([pred_rotmat[valid_boxes].view(valid_batch_size, -1), pred_betas[valid_boxes]],
                                        dim=1)
            loss_dict.update(
                {'pred_pose_shape': pred_pose_shape}
            )
        best_idxs = select_index(idxs_in_batch[valid_boxes], pose_idx[valid_boxes].int()[:, 0],
                                 bboxes_confidence[valid_boxes])
        sdf_loss = torch.zeros(len(best_idxs)).to(pred_vertices.device)
        for bid, ids in enumerate(best_idxs):
            if len(ids) <= 1:
                continue
            ids = torch.tensor(ids)
            sdf_loss[bid] = self.sdf_loss(pred_vertices[valid_boxes][ids], translation[valid_boxes][ids])
        loss_dict.update({
            'loss_sdf': sdf_loss.sum() if self.use_sdf else sdf_loss.sum().detach() * 1e-4
        })
        if self.nr_batch_rank:
            device = pred_vertices.device
            batch_rank_loss = torch.zeros(len(best_idxs)).to(pred_vertices.device)
            num_intruded_pixels = torch.zeros(len(best_idxs)).to(pred_vertices.device)
            erode_mask_loss = torch.zeros(len(best_idxs)).to(pred_vertices.device)

            K = torch.eye(3, device=device)
            K[0, 0] = K[1, 1] = self.FOCAL_LENGTH
            K[2, 2] = 1
            K[1, 2] = K[0, 2] = self.image_size / 2  # Because the neural renderer only support squared images
            K = K.unsqueeze(0)  # Our batch size is 1
            R = torch.eye(3, device=device).unsqueeze(0)
            t = torch.zeros(3, device=device).unsqueeze(0)

            for bid, ids in enumerate(best_idxs):
                if len(ids) <= 1 or scene[bid].max() < 1:
                    continue
                ids = torch.tensor(ids)
                verts = pred_vertices[valid_boxes][ids] + translation[valid_boxes][ids].unsqueeze(
                    1)  # num_personx6890x3
                cur_pose_idxs = pose_idx[valid_boxes][ids, 0]
                with torch.no_grad():
                    pose_idxs_int = cur_pose_idxs.int()
                    has_mask_gt = torch.zeros_like(pose_idxs_int)
                    for has_mask_idx, cur_pid in enumerate(pose_idxs_int):
                        has_mask_gt[has_mask_idx] = 1 if torch.sum(scene[bid] == (cur_pid + 1).item()) > 0 else 0

                if has_mask_gt.sum() < 1:
                    continue

                verts = verts[has_mask_gt > 0]
                cur_pose_idxs = cur_pose_idxs[has_mask_gt > 0]

                bs, nv = verts.shape[:2]
                face_tensor = torch.tensor(self.smpl.faces.astype(np.int64), dtype=torch.long,
                                           device=device).unsqueeze_(0).repeat([bs, 1, 1])
                bs, nf = face_tensor.shape[:2]
                textures = torch.ones_like(face_tensor).float() + cur_pose_idxs.to(device)[:, None, None]
                textures = textures[:, :, None, None, None, :]
                rgb, depth, mask = self.neural_renderer(verts, face_tensor.int(), textures=textures, K=K, R=R, t=t,
                                                        dist_coeffs=torch.tensor([[0., 0., 0., 0., 0.]], device=device))
                predicted_depth = depth[:, self.h_diff:rgb.shape[-2] - self.h_diff,
                                  self.w_diff:rgb.shape[-1] - self.w_diff]
                predicted_mask = mask[:, self.h_diff:rgb.shape[-2] - self.h_diff,
                                 self.w_diff:rgb.shape[-1] - self.w_diff]

                with torch.no_grad():
                    gt_foreground = scene[bid] > 0
                    foreground_select = (cur_pose_idxs.round().int() + 1)[:, None, None] == scene[bid].int()
                    intruded_parts_mask = torch.prod(predicted_mask, dim=0)
                    supervising_mask = intruded_parts_mask.unsqueeze(0).float() * gt_foreground.unsqueeze(
                        0).float() * (~foreground_select).float()
                    if supervising_mask.norm() == 0:  # No ordinal relationship errors is detected.
                        continue

                gt_closest_depth_multi = torch.zeros_like(predicted_depth)
                gt_closest_depth_multi[foreground_select] += predicted_depth[foreground_select]
                gt_closest_depth = gt_closest_depth_multi.sum(0)

                gt_closest_depth = gt_closest_depth.repeat([bs, 1, 1])
                ordinal_distance = (gt_closest_depth - predicted_depth) * supervising_mask
                penalize_ranks = torch.log(1. + torch.exp(ordinal_distance)) * supervising_mask
                # To avoid instable gradient:
                if torch.sum(ordinal_distance > 10) > 0:
                    penalize_ranks[ordinal_distance.detach() > 10] = ordinal_distance[ordinal_distance.detach() > 10]
                    print(f'{torch.sum(ordinal_distance > 10)} pixels found to be greater than 10 in batch rank loss')
                batch_rank_loss[bid] = penalize_ranks.mean()
                num_intruded_pixels[bid] = supervising_mask.sum()

            loss_dict.update({'loss_batch_rank': batch_rank_loss, 'num_intruded_pixels': num_intruded_pixels})

        if self.re_weight:
            for k, v in self.re_weight.items():
                if k.startswith('adv_loss'):
                    loss_dict[k] *= v
                else:
                    loss_dict[f'loss_{k}'] *= v

        return loss_dict

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d):
        """
        Compute 2D reprojection loss on the keypoints.
        The confidence is binary and indicates whether the keypoints exist or not.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d[:, -24:], gt_keypoints_2d[:, :, :-1]))
        return loss.mean(), loss.mean(dim=[1, 2]).detach()

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d):
        """
        Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence
        """
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        pred_keypoints_3d = pred_keypoints_3d[..., -24:, :]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            if hasattr(self, 'criterion_3d_keypoints'):
                return (conf * self.criterion_3d_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
            else:
                return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.tensor(0).float().cuda()

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl):
        """
        Compute per-vertex loss on the shape for the examples that SMPL annotations are available.
        """
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]
        if len(gt_vertices_with_shape) > 0:
            return self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return torch.tensor(0).float().cuda()

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        """
        Compute SMPL parameter loss for the examples that SMPL annotations are available.
        """
        batch_size = pred_rotmat.shape[0]
        pred_rotmat_valid = pred_rotmat[has_smpl == 1].view(-1, 3, 3)
        gt_rotmat_valid = batch_rodrigues(gt_pose[has_smpl == 1].view(-1, 3))
        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.tensor(0).float().cuda()
            loss_regr_betas = torch.tensor(0).float().cuda()
        return loss_regr_pose, loss_regr_betas

    def collision_loss(self, pred_vertices, best_idxs, loss_dict):
        """
        Calculate collision losses
        :param pred_vertices: Predicted vertices
        :param best_idxs: 2D list, each row means the selected meshes for this image
        :return:
        """
        device = pred_vertices.device
        max_persons = self.collision_param.get('max_persons', 16)
        loss = torch.zeros(len(best_idxs)).to(device)
        for bid, ids in enumerate(best_idxs):
            if len(ids) == 0:
                continue
            # Only select a limited number of persons to avoid OOM
            if len(ids) > max_persons:
                ids = random.sample(ids, max_persons)
            ids = torch.tensor(ids)

            verts = pred_vertices[ids]  # num_personx6890x3
            bs, nv = verts.shape[:2]
            face_tensor = torch.tensor(self.smpl.faces.astype(np.int64), dtype=torch.long,
                                       device=device).unsqueeze_(0).repeat([bs, 1, 1])
            bs, nf = face_tensor.shape[:2]
            faces_idx = face_tensor + (torch.arange(bs, dtype=torch.long).to(device) * nv)[:, None, None]
            # We cannot make a batch because the number of persons is not determined
            triangles = verts.view([-1, 3])[faces_idx.view([1, -1, 3])]

            with torch.no_grad():
                collision_idxs = self.search_tree(triangles)
                collision_idxs_ = torch.div(collision_idxs.clone(), nf)
                msk_self = (collision_idxs_[..., 0] == collision_idxs_[..., 1])
                collision_idxs[msk_self] = -1

            if collision_idxs.max() >= bs * nf:
                print(f'An overflow detected with bs: {bs}')
                continue

            cur_loss = self.pen_distance(triangles, collision_idxs).mean()

            if torch.isnan(cur_loss):
                print(f'A NaN is detected inside intersection loss with bs: {bs}')
                continue
            loss[bid] += cur_loss

        return loss.mean()

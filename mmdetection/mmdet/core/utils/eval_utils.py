import os.path as osp
import torch
import torchvision
from mmdetection.mmdet.models.utils.smpl_utils import batch_rodrigues, J24_TO_J14, H36M_TO_J14, J24_TO_H36M
from mmdetection.mmdet.models.utils.pose_utils import reconstruction_error, vectorize_distance
from mmdetection.mmdet.core.utils import AverageMeter
from mmdetection.mmdet.models.utils.camera import PerspectiveCamera
from mmdetection.mmdet.models.utils.smpl.renderer import Renderer
from mmdetection.mmdet.models.utils.smpl.body_models import SMPL, JointMapper
from mmdetection.mmdet.models.utils.smpl.viz import draw_skeleton, J24_TO_J14, get_bv_verts, plot_pose_H36M
import cv2
import matplotlib.pyplot as plt
import numpy as np
import abc
import math
import h5py
import scipy.io as scio
from sdf import CollisionVolume


def compute_scale_transform(S1, S2):
    '''
    Computes a scale transform (s, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    t 3x1 translation, s scale.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2
    # 2. Compute variance of X1,X2 used for scale.
    var1 = np.sum(X1 ** 2)
    var2 = np.sum(X2 ** 2)

    # 3. Recover scale.
    scale = np.sqrt(var2 / var1)

    # 4. Recover translation.
    t = mu2 - scale * (mu1)

    # 5. Error:
    S1_hat = scale * S1 + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_scale_transform_batch(S1, S2, visibility):
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i, visibility[i]] = compute_scale_transform(S1[i, visibility[i]], S2[i, visibility[i]])
    return S1_hat


def draw_text(input_image, content):
    """
    content is a dict. draws key: val on image
    Assumes key is str, val is float
    """
    image = input_image.copy()
    input_is_float = False
    if np.issubdtype(image.dtype, np.float):
        input_is_float = True
        image = (image * 255).astype(np.uint8)

    black = np.array([0, 0, 255])
    margin = 45
    start_x = 15
    start_y = margin
    for key in sorted(content.keys()):
        text = f"{key}: {content[key]}"
        image = cv2.putText(image, text, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        start_y += margin

    if input_is_float:
        image = image.astype(np.float32) / 255.
    return image


class EvalHandler(metaclass=abc.ABCMeta):
    def __init__(self, writer=print, log_every=50, viz_dir='', FOCAL_LENGTH=1000, work_dir=''):
        self.call_cnt = 0
        self.log_every = log_every
        self.writer = writer
        self.viz_dir = viz_dir
        self.work_dir = work_dir
        self.camera = PerspectiveCamera(FOCAL_LENGTH=FOCAL_LENGTH)
        self.FOCAL_LENGTH = FOCAL_LENGTH
        if self.viz_dir:
            self.renderer = Renderer(focal_length=FOCAL_LENGTH)
        else:
            self.renderer = None

    def __call__(self, *args, **kwargs):
        self.call_cnt += 1
        res = self.handle(*args, **kwargs)
        if self.log_every > 0 and (self.call_cnt % self.log_every == 0):
            self.log()
        return res

    @abc.abstractmethod
    def handle(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def log(self):
        pass

    def finalize(self):
        pass


class H36MEvalHandler(EvalHandler):

    def __init__(self, JOINT_REGRESSOR_H36M='data/J_regressor_h36m.npy', pattern='.60457274_', **kwargs):
        super().__init__(**kwargs)
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        self.pattern = pattern
        self.p1_meter = AverageMeter('P1', ':.2f')
        self.p2_meter = AverageMeter('P2', ':.2f')

    def handle(self, data_batch, pred_results, use_gt=False):
        pred_vertices = pred_results['pred_vertices'].cpu()

        gt_keypoints_3d = data_batch['gt_kpts3d'].data[0][0].clone().repeat([pred_vertices.shape[0], 1, 1])
        gt_pelvis_smpl = gt_keypoints_3d[:, [14], :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[:, J24_TO_J14, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis_smpl

        J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(
            pred_vertices.device)
        # Get 14 predicted joints from the SMPL mesh
        pred_keypoints_3d_smpl = torch.matmul(J_regressor_batch, pred_vertices)
        pred_pelvis_smpl = pred_keypoints_3d_smpl[:, [0], :].clone()
        pred_keypoints_3d_smpl = pred_keypoints_3d_smpl[:, H36M_TO_J14, :]
        pred_keypoints_3d_smpl = pred_keypoints_3d_smpl - pred_pelvis_smpl

        file_name = data_batch['img_meta'].data[0][0]['file_name']

        # Compute error metrics
        # Absolute error (MPJPE)
        error_smpl = torch.sqrt(((pred_keypoints_3d_smpl - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(
            dim=-1)

        mpjpe = float(error_smpl.min() * 1000)
        self.p1_meter.update(mpjpe)

        if self.pattern in file_name:
            # Reconstruction error
            r_error_smpl = reconstruction_error(pred_keypoints_3d_smpl.cpu().numpy(), gt_keypoints_3d.cpu().numpy(),
                                                reduction=None)
            r_error = float(r_error_smpl.min() * 1000)
            self.p2_meter.update(r_error)
        else:
            r_error = -1

        save_pack = {'file_name': file_name,
                     'MPJPE': mpjpe,
                     'r_error': r_error,
                     'pred_rotmat': pred_results['pred_rotmat'],
                     'pred_betas': pred_results['pred_betas'],
                     }
        return save_pack

    def log(self):
        self.writer(f'p1: {self.p1_meter.avg:.2f}mm, p2: {self.p2_meter.avg:.2f}mm')


class PanopticEvalHandler(EvalHandler):

    def __init__(self, JOINT_REGRESSOR_H36M='data/J_regressor_h36m.npy', **kwargs):
        super().__init__(**kwargs)
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        self.p1_meter = AverageMeter('P1', ':.2f')
        self.stats = list()
        self.mismatch_cnt = 0
        # Initialize SMPL model
        openpose_joints = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                           7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
        extra_joints = [8, 5, 45, 46, 4, 7, 21, 19, 17, 16, 18, 20, 47, 48, 49, 50, 51, 52, 53, 24, 26, 25, 28, 27]
        joints = torch.tensor(openpose_joints + extra_joints, dtype=torch.int32)
        joint_mapper = JointMapper(joints)
        smpl_params = dict(model_folder='data/smpl',
                           joint_mapper=joint_mapper,
                           create_glb_pose=True,
                           body_pose_param='identity',
                           create_body_pose=True,
                           create_betas=True,
                           # create_trans=True,
                           dtype=torch.float32,
                           vposer_ckpt=None,
                           gender='neutral')
        self.smpl = SMPL(**smpl_params)
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        self.collision_meter = AverageMeter('P3', ':.2f')
        self.collision_volume = CollisionVolume(self.smpl.faces, grid_size=64).cuda()
        self.coll_cnt = 0
        self.threshold_list = [0.1, 0.15, 0.2]
        self.total_ordinal_cnt = {i: 0 for i in self.threshold_list}
        self.correct_ordinal_cnt = {i: 0 for i in self.threshold_list}

    def handle(self, data_batch, pred_results, use_gt=False):
        # Evaluate collision metric
        pred_vertices = pred_results['pred_vertices']
        pred_translation = pred_results['pred_translation']
        cur_collision_volume = self.collision_volume(pred_vertices, pred_translation)
        if cur_collision_volume.item() > 0:
            # self.writer(f'Collision found with {cur_collision_volume.item() * 1000} L')
            self.coll_cnt += 1
        self.collision_meter.update(cur_collision_volume.item() * 1000.)

        pred_vertices = pred_results['pred_vertices'].cpu()
        pred_camera = pred_results['pred_camera'].cpu()
        pred_translation = pred_results['pred_translation'].cpu()
        bboxes = pred_results['bboxes'][0][:, :4]
        img = data_batch['img'].data[0][0].clone()

        gt_keypoints_3d = data_batch['gt_kpts3d'].data[0][0].clone()
        gt_pelvis_smpl = gt_keypoints_3d[:, [14], :-1].clone()
        visible_kpts = gt_keypoints_3d[:, J24_TO_H36M, -1].clone()
        origin_gt_kpts3d = data_batch['gt_kpts3d'].data[0][0].clone().cpu()
        origin_gt_kpts3d = origin_gt_kpts3d[:, J24_TO_H36M]
        # origin_gt_kpts3d[:, :, :-1] -= gt_pelvis_smpl
        gt_keypoints_3d = gt_keypoints_3d[:, J24_TO_H36M, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis_smpl

        J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(
            pred_vertices.device)
        # Get 14 predicted joints from the SMPL mesh
        pred_keypoints_3d_smpl = torch.matmul(J_regressor_batch, pred_vertices)
        pred_pelvis_smpl = pred_keypoints_3d_smpl[:, [0], :].clone()
        # pred_keypoints_3d_smpl = pred_keypoints_3d_smpl[:, H36M_TO_J14, :]
        pred_keypoints_3d_smpl = pred_keypoints_3d_smpl - pred_pelvis_smpl

        file_name = data_batch['img_meta'].data[0][0]['file_name']
        fname = osp.basename(file_name)

        # To select closest points
        glb_vis = (visible_kpts.sum(0) >= (
                visible_kpts.shape[0] - 0.1)).float()[None, :, None]  # To avoid in-accuracy in float point number
        if use_gt:
            paired_idxs = torch.arange(gt_keypoints_3d.shape[0])
        else:
            dist = vectorize_distance((glb_vis * gt_keypoints_3d).numpy(),
                                      (glb_vis * pred_keypoints_3d_smpl).numpy())
            paired_idxs = torch.from_numpy(dist.argmin(1))
        is_mismatch = len(set(paired_idxs.tolist())) < len(paired_idxs)
        if is_mismatch:
            self.mismatch_cnt += 1

        selected_prediction = pred_keypoints_3d_smpl[paired_idxs]

        # Compute error metrics
        # Absolute error (MPJPE)
        error_smpl = (torch.sqrt(((selected_prediction - gt_keypoints_3d) ** 2).sum(dim=-1)) * visible_kpts)

        mpjpe = float(error_smpl.mean() * 1000)
        self.p1_meter.update(mpjpe, n=error_smpl.shape[0])

        save_pack = {'file_name': osp.basename(file_name),
                     'MPJPE': mpjpe,
                     'pred_rotmat': pred_results['pred_rotmat'].cpu(),
                     'pred_betas': pred_results['pred_betas'].cpu(),
                     'gt_kpts': origin_gt_kpts3d,
                     'kpts_paired': selected_prediction,
                     'pred_kpts': pred_keypoints_3d_smpl,
                     }

        if self.viz_dir and (is_mismatch or error_smpl.mean(-1).min() * 1000 > 200):
            img = img.clone() * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor(
                [0.485, 0.456, 0.406]).view(3, 1, 1)
            img_cv = img.clone().numpy()
            img_cv = (img_cv * 255).astype(np.uint8).transpose([1, 2, 0]).copy()
            for bbox in bboxes[paired_idxs]:
                img_cv = cv2.rectangle(img_cv, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            img_cv = draw_text(img_cv, {'mismatch': is_mismatch, 'error': str(error_smpl.mean(-1) * 1000)});
            img_cv = (img_cv / 255.)

            torch.set_printoptions(precision=1)
            img_render = self.renderer([torch.tensor(img_cv.transpose([2, 0, 1]))], [pred_vertices],
                                       translation=[pred_translation])

            bv_verts = get_bv_verts(bboxes, pred_vertices, pred_translation,
                                    img.shape, self.FOCAL_LENGTH)
            img_bv = self.renderer([torch.ones_like(img)], [bv_verts],
                                   translation=[torch.zeros(bv_verts.shape[0], 3)])
            img_grid = torchvision.utils.make_grid(torch.tensor(([img_render[0], img_bv[0]])),
                                                   nrow=2).numpy().transpose([1, 2, 0])
            img_grid[img_grid > 1] = 1
            img_grid[img_grid < 0] = 0
            plt.imsave(osp.join(self.viz_dir, fname), img_grid)
        return save_pack

    def log(self):
        self.writer(
            f'p1: {self.p1_meter.avg:.2f}mm, coll_cnt: {self.coll_cnt} coll: {self.collision_meter.avg} L')


class MuPoTSEvalHandler(EvalHandler):

    def __init__(self, JOINT_REGRESSOR_H36M='data/J_regressor_h36m.npy', **kwargs):
        super().__init__(**kwargs)
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        self.p1_meter = AverageMeter('P1', ':.2f')
        self.p2_meter = AverageMeter('P2', ':.2f')
        self.p3_meter = AverageMeter('P3', ':.2f')
        self.stats = list()
        self.mismatch_cnt = 0
        # Initialize SMPL model
        openpose_joints = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                           7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
        extra_joints = [8, 5, 45, 46, 4, 7, 21, 19, 17, 16, 18, 20, 47, 48, 49, 50, 51, 52, 53, 24, 26, 25, 28, 27]
        joints = torch.tensor(openpose_joints + extra_joints, dtype=torch.int32)
        joint_mapper = JointMapper(joints)
        smpl_params = dict(model_folder='data/smpl',
                           joint_mapper=joint_mapper,
                           create_glb_pose=True,
                           body_pose_param='identity',
                           create_body_pose=True,
                           create_betas=True,
                           # create_trans=True,
                           dtype=torch.float32,
                           vposer_ckpt=None,
                           gender='neutral')
        self.smpl = SMPL(**smpl_params)
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        self.result_list = list()
        self.result_list_2d = list()
        self.h36m_to_MPI = [10, 8, 14, 15, 16, 11, 12, 13, 4, 5, 6, 1, 2, 3, 0, 7, 9]

        self.collision_meter = AverageMeter('collision', ':.2f')
        self.collision_volume = CollisionVolume(self.smpl.faces, grid_size=64).cuda()
        self.coll_cnt = 0

    def handle(self, data_batch, pred_results, use_gt=False):
        # Evaluate collision metric
        pred_vertices = pred_results['pred_vertices']
        pred_translation = pred_results['pred_translation']
        cur_collision_volume = self.collision_volume(pred_vertices, pred_translation)
        if cur_collision_volume.item() > 0:
            # self.writer(f'Collision found with {cur_collision_volume.item() * 100 } L')
            self.coll_cnt += 1
        self.collision_meter.update(cur_collision_volume.item() * 1000.)

        pred_vertices = pred_results['pred_vertices'].cpu()
        pred_camera = pred_results['pred_camera'].cpu()
        pred_translation = pred_results['pred_translation'].cpu()
        bboxes = pred_results['bboxes'][0][:, :4]
        img = data_batch['img'].data[0][0].clone()

        J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(
            pred_vertices.device)
        # Get 14 predicted joints from the SMPL mesh
        pred_keypoints_3d_smpl = torch.matmul(J_regressor_batch, pred_vertices)
        pred_pelvis_smpl = pred_keypoints_3d_smpl[:, [0], :].clone()
        self.result_list.append(
            (pred_keypoints_3d_smpl[:, self.h36m_to_MPI] + pred_translation[:, None]).numpy())
        batch_size = pred_keypoints_3d_smpl.shape[0]
        img_size = torch.zeros(batch_size, 2).to(pred_keypoints_3d_smpl.device)
        img_size += torch.tensor(img.shape[:-3:-1], dtype=img_size.dtype).to(img_size.device)
        rotation_Is = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(pred_keypoints_3d_smpl.device)
        pred_keypoints_2d_smpl = self.camera(pred_keypoints_3d_smpl, batch_size=batch_size, rotation=rotation_Is,
                                             translation=pred_translation,
                                             center=img_size / 2)
        if self.viz_dir:
            img = img.clone() * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor(
                [0.485, 0.456, 0.406]).view(3, 1, 1)
            img_cv = img.clone().numpy()
            img_cv = (img_cv * 255).astype(np.uint8).transpose([1, 2, 0]).copy()
            for kpts, bbox in zip(pred_keypoints_2d_smpl.numpy(), bboxes):
                img_cv = cv2.rectangle(img_cv, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                img_cv = draw_skeleton(img_cv, kpts[H36M_TO_J14, :2])
            # img_cv = draw_text(img_cv, {'mismatch': is_mismatch, 'error': str(error_smpl.mean(-1) * 1000)});
            img_cv = (img_cv / 255.)
            fname = osp.basename(data_batch['img_meta'].data[0][0]['file_name'])
            plt.imsave(osp.join(self.viz_dir, fname), img_cv)

        scale_factor = data_batch['img_meta'].data[0][0]['scale_factor']
        raw_kpts2d = pred_keypoints_2d_smpl / scale_factor
        self.result_list_2d.append(raw_kpts2d[:, self.h36m_to_MPI])
        return {'file_name': data_batch['img_meta'].data[0][0]['file_name'], 'pred_kpts3d': pred_keypoints_3d_smpl}

    def log(self):
        self.writer(f'coll_cnt: {self.coll_cnt} coll {self.collision_meter.avg} L')

    def finalize(self):
        max_persons = max([i.shape[0] for i in self.result_list])
        result = np.zeros((len(self.result_list), max_persons, 17, 3))
        result_2d = np.zeros((len(self.result_list), max_persons, 17, 2))
        for i, (r, r_2d) in enumerate(zip(self.result_list, self.result_list_2d)):
            result[i, :r.shape[0]] = r
            result_2d[i, :r.shape[0]] = r_2d
        scio.savemat(osp.join(self.work_dir, 'mupots.mat'), {'result': result, 'result_2d': result_2d})

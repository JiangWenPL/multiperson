import torch
from torch import nn
import numpy as np
import h5py
from mmcv.cnn import kaiming_init, normal_init

from ..registry import HEADS

from ..utils.smpl.smpl import SMPL
from ..utils.smpl_utils import rot6d_to_rotmat, batch_rodrigues
from ..builder import build_loss
from .smpl_common import get_smpl_target_single, dp_get_smpl_target_single


@HEADS.register_module
class SMPLHead(nn.Module):

    def __init__(self, in_size=7, in_channels=256, num_convs=4, conv_out_channels=256,
                 init_param_file='data/neutral_smpl_mean_params.h5',
                 joint_names=None, joint_map=None, joint_regressor_extra=None, FOCAL_LENGTH=1000,
                 loss_cfg=dict(type='SMPLLoss'), implicity_size=1024,
                 ):
        super(SMPLHead, self).__init__()

        # Load SMPL mean parameters
        f = h5py.File(init_param_file, 'r')
        init_grot = np.array([np.pi, 0., 0.])
        init_pose = np.hstack([init_grot, f['pose'][3:]])
        init_pose = torch.tensor(init_pose.astype('float32'))
        init_rotmat = batch_rodrigues(init_pose.contiguous().view(-1, 3))
        init_contrep = init_rotmat.view(-1, 3, 3)[:, :, ::2].contiguous().view(-1)

        init_shape = torch.tensor(f['shape'][:].astype('float32'))
        init_cam = torch.tensor([0.9, 0, 0])

        self.register_buffer('init_contrep', init_contrep)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)
        self.npose = init_rotmat.shape[0] * 6
        # Multiply by 6 as we need to estimate two matrixes on each joins

        self.conv_out_channels = conv_out_channels
        self.convs = nn.ModuleList()
        for i in range(num_convs):
            # NOTE: no need for concatenation of mask feature and mask prediction
            in_channels = self.conv_out_channels if i > 0 else in_channels
            stride = 2 if i == num_convs - 1 else 1
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        self.conv_out_channels,
                        3,
                        stride=stride,
                        padding=1),
                    nn.BatchNorm2d(self.conv_out_channels)))

        self.avgpool = nn.AvgPool2d(in_size // 2, stride=1)

        self.implicity_size = implicity_size

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.conv_out_channels + 2 * 72 + 10 + 3, self.implicity_size)

        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(self.implicity_size, self.implicity_size)

        self.drop2 = nn.Dropout()
        self.dec = nn.Linear(self.implicity_size, self.npose + 10 + 3)

        # Initialize SMPL model
        self.smpl = SMPL('data/smpl')
        self.FOCAL_LENGTH = FOCAL_LENGTH
        self.loss = build_loss(loss_cfg)

    def forward(self, x):
        batch_size = x.shape[0]

        init_pose = self.init_contrep.view(1, -1).expand(batch_size, -1)
        init_shape = self.init_shape.view(1, -1).expand(batch_size, -1)
        init_cam = self.init_cam.view(1, -1).expand(batch_size, -1)

        for conv in self.convs:
            x = self.relu(conv(x))

        xf = self.avgpool(x)
        xf = xf.view(xf.size(0), -1)

        x0 = torch.cat([init_pose, init_shape, init_cam], 1)
        xf1 = torch.cat([xf, x0], 1)
        xf1 = self.fc1(xf1)
        xf1 = self.relu(xf1)
        xf1 = self.drop1(xf1)
        xf1 = self.fc2(xf1)
        xf1 = self.relu(xf1)
        xf1 = self.drop2(xf1)
        x1 = self.dec(xf1) + x0
        xpose1 = x1[:, :self.npose].contiguous()
        xshape1 = x1[:, self.npose:self.npose + 10].contiguous()
        xcam1 = x1[:, self.npose + 10:].contiguous()

        xf2 = torch.cat([xf, x1], 1)
        xf2 = self.fc1(xf2)
        xf2 = self.relu(xf2)
        xf2 = self.drop1(xf2)
        xf2 = self.fc2(xf2)
        xf2 = self.relu(xf2)
        xf2 = self.drop2(xf2)
        x2 = self.dec(xf2) + x1
        xpose2 = x2[:, :self.npose].contiguous()
        xshape2 = x2[:, self.npose:self.npose + 10].contiguous()
        xcam2 = x2[:, self.npose + 10:].contiguous()

        xf3 = torch.cat([xf, x2], 1)
        xf3 = self.fc1(xf3)
        xf3 = self.relu(xf3)
        xf3 = self.drop1(xf3)
        xf3 = self.fc2(xf3)
        xf3 = self.relu(xf3)
        xf3 = self.drop2(xf3)
        x3 = self.dec(xf3) + x2

        xpose3 = x3[:, :self.npose].contiguous()
        xshape3 = x3[:, self.npose:self.npose + 10].contiguous()
        xcam3 = x3[:, self.npose + 10:].contiguous()

        pred_camera = xcam3
        pred_betas = xshape3
        pred_rotmat = rot6d_to_rotmat(xpose3).view(batch_size, 24, 3, 3)

        smpl_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
        pred_vertices = smpl_output.vertices
        pred_joints = smpl_output.joints

        return {'pred_rotmat': pred_rotmat, 'pred_betas': pred_betas, 'pred_camera': pred_camera,
                'pred_vertices': pred_vertices, 'pred_joints': pred_joints}

    def get_target(self, sampling_results, gt_kpts2d, gt_kpts3d, gt_poses, gt_shapes, gt_trans, has_smpl,
                   rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        cfg_list = [rcnn_train_cfg for _ in range(len(pos_proposals))]
        the_map = map(
            get_smpl_target_single, [self] * len(pos_assigned_gt_inds), pos_proposals, pos_assigned_gt_inds, gt_kpts2d,
            gt_kpts3d, gt_poses, gt_shapes,
            gt_trans, has_smpl, cfg_list, range(len(pos_assigned_gt_inds)))
        return tuple(map(torch.cat, zip(*the_map)))

    def get_dp_target(self, sampling_results, gt_kpts2d, gt_kpts3d, gt_poses, gt_shapes, gt_trans, has_smpl,
                      rcnn_train_cfg, dp_x, dp_y, dp_U, dp_V, dp_I, dp_num_pts):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        cfg_list = [rcnn_train_cfg for _ in range(len(pos_proposals))]
        the_map = map(
            dp_get_smpl_target_single, [self] * len(pos_assigned_gt_inds), pos_proposals, pos_assigned_gt_inds,
            gt_kpts2d,
            gt_kpts3d, gt_poses, gt_shapes,
            gt_trans, has_smpl, cfg_list, range(len(pos_assigned_gt_inds)),
            dp_x, dp_y, dp_U, dp_V, dp_I, dp_num_pts
        )
        return tuple(map(torch.cat, zip(*the_map)))

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.dec.weight, gain=0.1)
        nn.init.zeros_(self.dec.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        for conv in self.convs:
            for sub_layer in conv:
                if isinstance(sub_layer, nn.Conv2d):
                    kaiming_init(sub_layer)
                elif isinstance(sub_layer, nn.BatchNorm2d):
                    nn.init.constant_(sub_layer.weight, 1)
                    nn.init.constant_(sub_layer.bias, 0)
                else:
                    raise RuntimeError("Unknown network component in SMPL Head")

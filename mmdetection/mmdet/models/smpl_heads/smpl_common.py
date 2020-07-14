import torch
import torch.nn as nn
import numpy as np
import sys
import torch
import numpy as np


def get_smpl_target_single(self, pos_proposals, pos_assigned_gt_inds, gt_kpts2d, gt_kpts3d, gt_poses, gt_shapes,
                           gt_trans, has_smpl, cfg, idx_in_batch):
    num_pos = pos_proposals.size(0)
    bboxs_proposal, kpts2d_target, kpts3d_target, poses_target, shapes_target, trans_target, has_smpl_target, \
    vertices_targets, idxs_in_batch, pose_idx = list(), list(), list(), list(), list(), list(), list(), list(), list(), list()
    if num_pos > 0:
        smpl_output = self.smpl(betas=gt_shapes, body_pose=gt_poses[:, 1:],
                                global_orient=gt_poses[:, None, 0], pose2rot=True)
        vertices = smpl_output.vertices
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        for i in range(num_pos):
            bbox = pos_proposals[i, :]
            idx = pos_assigned_gt_inds[i]
            # x1, y1, x2, y2 = bbox
            # w = np.maximum(x2 - x1 + 1, 1)
            # h = np.maximum(y2 - y1 + 1, 1)
            bboxs_proposal.append(bbox)
            kpts2d_target.append(gt_kpts2d[idx])
            kpts3d_target.append(gt_kpts3d[idx])
            poses_target.append(gt_poses[idx])
            shapes_target.append(gt_shapes[idx])
            trans_target.append(gt_trans[idx])
            has_smpl_target.append(has_smpl[idx])
            vertices_targets.append(vertices[idx])
            idxs_in_batch.append(torch.tensor([idx_in_batch]))
            pose_idx.append(torch.tensor([idx]))
    else:
        pass

    # To avoid duplicate declartion
    return tuple(map(
        lambda x: torch.stack(x).float().to(pos_proposals.device),
        [bboxs_proposal, kpts2d_target, kpts3d_target, poses_target, shapes_target, trans_target, has_smpl_target,
         vertices_targets, idxs_in_batch, pose_idx
         ]))


def dp_get_smpl_target_single(self, pos_proposals, pos_assigned_gt_inds, gt_kpts2d, gt_kpts3d, gt_poses, gt_shapes,
                              gt_trans, has_smpl, cfg, idx_in_batch, dp_x, dp_y, dp_U, dp_V, dp_I, dp_num_pts):
    num_pos = pos_proposals.size(0)
    bboxs_proposal, kpts2d_target, kpts3d_target, poses_target, shapes_target, trans_target, has_smpl_target, \
    vertices_targets, idxs_in_batch, pose_idx, dp_x_target, dp_y_target, dp_U_target, dp_V_target, dp_I_target, dp_num_pts_target = \
        list(), list(), list(), list(), list(), \
        list(), list(), list(), list(), list(), \
        list(), list(), list(), list(), list(), list()
    if num_pos > 0:
        smpl_output = self.smpl(betas=gt_shapes, body_pose=gt_poses[:, 1:],
                                global_orient=gt_poses[:, None, 0], pose2rot=True)
        vertices = smpl_output.vertices
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        for i in range(num_pos):
            bbox = pos_proposals[i, :]
            idx = pos_assigned_gt_inds[i]
            # x1, y1, x2, y2 = bbox
            # w = np.maximum(x2 - x1 + 1, 1)
            # h = np.maximum(y2 - y1 + 1, 1)
            bboxs_proposal.append(bbox)
            kpts2d_target.append(gt_kpts2d[idx])
            kpts3d_target.append(gt_kpts3d[idx])
            poses_target.append(gt_poses[idx])
            shapes_target.append(gt_shapes[idx])
            trans_target.append(gt_trans[idx])
            has_smpl_target.append(has_smpl[idx])
            vertices_targets.append(vertices[idx])
            idxs_in_batch.append(torch.tensor([idx_in_batch]))
            pose_idx.append(torch.tensor([idx]))
            dp_x_target.append(dp_x[idx])
            dp_y_target.append(dp_y[idx])
            dp_U_target.append(dp_U[idx])
            dp_V_target.append(dp_V[idx])
            dp_I_target.append(dp_I[idx])
            dp_num_pts_target.append(dp_num_pts[idx])
    else:
        pass

    # To avoid duplicate declartion
    return tuple(map(
        lambda x: torch.stack(x).float().to(pos_proposals.device),
        [bboxs_proposal, kpts2d_target, kpts3d_target, poses_target, shapes_target, trans_target, has_smpl_target,
         vertices_targets, idxs_in_batch, pose_idx,
         dp_x_target, dp_y_target, dp_U_target, dp_V_target, dp_I_target, dp_num_pts_target
         ]))

'''
Code for discriminator is taken from
https://github.com/MandyMo/pytorch_HMR/blob/master/src/Discriminator.py
'''

class LinearModel(nn.Module):
    '''
        input param:
            fc_layers: a list of neuron count, such as [2133, 1024, 1024, 85]
            use_dropout: a list of bool define use dropout or not for each layer, such as [True, True, False]
            drop_prob: a list of float defined the drop prob, such as [0.5, 0.5, 0]
            use_ac_func: a list of bool define use active function or not, such as [True, True, False]
    '''

    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func):
        super(LinearModel, self).__init__()
        self.fc_layers = fc_layers
        self.use_dropout = use_dropout
        self.drop_prob = drop_prob
        self.use_ac_func = use_ac_func

        if not self._check():
            msg = 'wrong LinearModel parameters!'
            print(msg)
            sys.exit(msg)

        self.create_layers()

    def _check(self):
        while True:
            if not isinstance(self.fc_layers, list):
                print('fc_layers require list, get {}'.format(type(self.fc_layers)))
                break

            if not isinstance(self.use_dropout, list):
                print('use_dropout require list, get {}'.format(type(self.use_dropout)))
                break

            if not isinstance(self.drop_prob, list):
                print('drop_prob require list, get {}'.format(type(self.drop_prob)))
                break

            if not isinstance(self.use_ac_func, list):
                print('use_ac_func require list, get {}'.format(type(self.use_ac_func)))
                break

            l_fc_layer = len(self.fc_layers)
            l_use_drop = len(self.use_dropout)
            l_drop_porb = len(self.drop_prob)
            l_use_ac_func = len(self.use_ac_func)

            return l_fc_layer >= 2 and l_use_drop < l_fc_layer and l_drop_porb < l_fc_layer and l_use_ac_func < l_fc_layer and l_drop_porb == l_use_drop

        return False

    def create_layers(self):
        l_fc_layer = len(self.fc_layers)
        l_use_drop = len(self.use_dropout)
        l_drop_porb = len(self.drop_prob)
        l_use_ac_func = len(self.use_ac_func)

        self.fc_blocks = nn.Sequential()

        for _ in range(l_fc_layer - 1):
            self.fc_blocks.add_module(
                name='regressor_fc_{}'.format(_),
                module=nn.Linear(in_features=self.fc_layers[_], out_features=self.fc_layers[_ + 1])
            )

            if _ < l_use_ac_func and self.use_ac_func[_]:
                self.fc_blocks.add_module(
                    name='regressor_af_{}'.format(_),
                    module=nn.ReLU()
                )

            if _ < l_use_drop and self.use_dropout[_]:
                self.fc_blocks.add_module(
                    name='regressor_fc_dropout_{}'.format(_),
                    module=nn.Dropout(p=self.drop_prob[_])
                )

    def forward(self, inputs):
        msg = 'the base class [LinearModel] is not callable!'
        sys.exit(msg)


'''
    shape discriminator is used for shape discriminator
    the inputs if N x 10
'''


class ShapeDiscriminator(LinearModel):
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func):
        if fc_layers[-1] != 1:
            msg = 'the neuron count of the last layer must be 1, but got {}'.format(fc_layers[-1])
            sys.exit(msg)

        super(ShapeDiscriminator, self).__init__(fc_layers, use_dropout, drop_prob, use_ac_func)

    def forward(self, inputs):
        return self.fc_blocks(inputs)


class PoseDiscriminator(nn.Module):
    def __init__(self, channels):
        super(PoseDiscriminator, self).__init__()

        if channels[-1] != 1:
            msg = 'the neuron count of the last layer must be 1, but got {}'.format(channels[-1])
            sys.exit(msg)

        self.conv_blocks = nn.Sequential()
        l = len(channels)
        for idx in range(l - 2):
            self.conv_blocks.add_module(
                name='conv_{}'.format(idx),
                module=nn.Conv2d(in_channels=channels[idx], out_channels=channels[idx + 1], kernel_size=1, stride=1)
            )

        self.fc_layer = nn.ModuleList()
        for idx in range(23):
            self.fc_layer.append(nn.Linear(in_features=channels[l - 2], out_features=1))

    # N x 23 x 9
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        inputs = inputs.transpose(1, 2).unsqueeze(2)  # to N x 9 x 1 x 23
        internal_outputs = self.conv_blocks(inputs)  # to N x c x 1 x 23
        o = []
        for idx in range(23):
            o.append(self.fc_layer[idx](internal_outputs[:, :, 0, idx]))

        return torch.cat(o, 1), internal_outputs


class FullPoseDiscriminator(LinearModel):
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func):
        if fc_layers[-1] != 1:
            msg = 'the neuron count of the last layer must be 1, but got {}'.format(fc_layers[-1])
            sys.exit(msg)

        super(FullPoseDiscriminator, self).__init__(fc_layers, use_dropout, drop_prob, use_ac_func)

    def forward(self, inputs):
        return self.fc_blocks(inputs)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self._read_configs()

        self._create_sub_modules()

    def _read_configs(self):
        self.beta_count = 10  # args.beta_count
        self.smpl_model = 0  # args.smpl_model
        self.smpl_mean_theta_path = 0  # args.smpl_mean_theta_path
        self.total_theta_count = 226  # args.total_theta_count
        self.joint_count = 24  # args.joint_count
        self.feature_count = 2048  # args.feature_count

    def _create_sub_modules(self):
        '''
            create theta discriminator for 23 joint
        '''

        self.pose_discriminator = PoseDiscriminator([9, 32, 32, 1])

        '''
            create full pose discriminator for total 23 joints
        '''
        fc_layers = [23 * 32, 1024, 1024, 1]
        use_dropout = [False, False, False]
        drop_prob = [0.5, 0.5, 0.5]
        use_ac_func = [True, True, False]
        self.full_pose_discriminator = FullPoseDiscriminator(fc_layers, use_dropout, drop_prob, use_ac_func)

        '''
            shape discriminator for betas
        '''
        fc_layers = [self.beta_count, 5, 1]
        use_dropout = [False, False]
        drop_prob = [0.5, 0.5]
        use_ac_func = [True, False]
        self.shape_discriminator = ShapeDiscriminator(fc_layers, use_dropout, drop_prob, use_ac_func)

    '''
        inputs is N x 226 (216 + 10)
    '''

    def forward(self, thetas):
        batch_size = thetas.shape[0]
        poses, shapes = thetas[:, :216], thetas[:, 216:]
        shape_disc_value = self.shape_discriminator(shapes)
        rotate_matrixs = poses.contiguous().view(-1, 24, 9)[:, 1:, :]
        pose_disc_value, pose_inter_disc_value = self.pose_discriminator(rotate_matrixs)
        full_pose_disc_value = self.full_pose_discriminator(pose_inter_disc_value.contiguous().view(batch_size, -1))
        return torch.cat((pose_disc_value, full_pose_disc_value, shape_disc_value), 1)


if __name__ == '__main__':
    device = torch.device('cuda')
    net = Discriminator()
    inputs = torch.ones((100, 226))
    disc_value = net(inputs)
    print(net)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


def draw_skeleton(input_image, joints, draw_edges=True, vis=None, radius=None):
    """
    joints is 3 x 19. but if not will transpose it.
    0: Right ankle
    1: Right knee
    2: Right hip
    3: Left hip
    4: Left knee
    5: Left ankle
    6: Right wrist
    7: Right elbow
    8: Right shoulder
    9: Left shoulder
    10: Left elbow
    11: Left wrist
    12: Neck
    13: Head top
    14: nose
    15: left_eye
    16: right_eye
    17: left_ear
    18: right_ear
    """

    if radius is None:
        radius = max(4, (np.mean(input_image.shape[:2]) * 0.01).astype(int))

    colors = {
        'pink': np.array([197, 27, 125]),  # L lower leg
        'light_pink': np.array([233, 163, 201]),  # L upper leg
        'light_green': np.array([161, 215, 106]),  # L lower arm
        'green': np.array([77, 146, 33]),  # L upper arm
        'red': np.array([215, 48, 39]),  # head
        'light_red': np.array([252, 146, 114]),  # head
        'light_orange': np.array([252, 141, 89]),  # chest
        'purple': np.array([118, 42, 131]),  # R lower leg
        'light_purple': np.array([175, 141, 195]),  # R upper
        'light_blue': np.array([145, 191, 219]),  # R lower arm
        'blue': np.array([69, 117, 180]),  # R upper arm
        'gray': np.array([130, 130, 130]),  #
        'white': np.array([255, 255, 255]),  #
    }
    colors = {k: tuple(int(i) for i in v) for k, v in colors.items()}

    image = input_image.copy()
    input_is_float = False

    if np.issubdtype(image.dtype, np.float):
        input_is_float = True
        max_val = image.max()
        if max_val <= 2.:  # should be 1 but sometimes it's slightly above 1
            image = (image * 255).astype(np.uint8)
        else:
            image = (image).astype(np.uint8)

    if joints.shape[0] != 2:
        joints = joints.T
    joints = np.round(joints).astype(int)

    jcolors = [
        'light_pink', 'light_pink', 'light_pink', 'pink', 'pink', 'pink',
        'light_blue', 'light_blue', 'light_blue', 'blue', 'blue', 'blue',
        'purple', 'purple', 'red', 'green', 'green', 'white', 'white'
    ]

    if joints.shape[1] == 19:
        # parent indices -1 means no parents
        parents = np.array([
            1, 2, 8, 9, 3, 4, 7, 8, 12, 12, 9, 10, 14, -1, 13, -1, -1, 15, 16
        ])
        # Left is light and right is dark
        ecolors = {
            0: 'light_pink',
            1: 'light_pink',
            2: 'light_pink',
            3: 'pink',
            4: 'pink',
            5: 'pink',
            6: 'light_blue',
            7: 'light_blue',
            8: 'light_blue',
            9: 'blue',
            10: 'blue',
            11: 'blue',
            12: 'purple',
            17: 'light_green',
            18: 'light_green',
            14: 'purple'
        }
    elif joints.shape[1] == 14:
        parents = np.array([
            1,
            2,
            8,
            9,
            3,
            4,
            7,
            8,
            -1,
            -1,
            9,
            10,
            13,
            -1,
        ])
        ecolors = {
            0: 'light_pink',
            1: 'light_pink',
            2: 'light_pink',
            3: 'pink',
            4: 'pink',
            5: 'pink',
            6: 'light_blue',
            7: 'light_blue',
            10: 'light_blue',
            11: 'blue',
            12: 'purple'
        }
    else:
        print('Unknown skeleton!!')

    for child in range(len(parents)):
        point = joints[:, child]
        # If invisible skip
        if vis is not None and vis[child] == 0:
            continue
        if draw_edges:
            cv2.circle(image, (point[0], point[1]), radius, colors['white'],
                       -1)
            cv2.circle(image, (point[0], point[1]), radius - 1,
                       colors[jcolors[child]], -1)
        else:
            # cv2.circle(image, (point[0], point[1]), 5, colors['white'], 1)
            cv2.circle(image, (point[0], point[1]), radius - 1,
                       colors[jcolors[child]], 1)
            # cv2.circle(image, (point[0], point[1]), 5, colors['gray'], -1)
        pa_id = parents[child]
        if draw_edges and pa_id >= 0:
            if vis is not None and vis[pa_id] == 0:
                continue
            point_pa = joints[:, pa_id]
            cv2.circle(image, (point_pa[0], point_pa[1]), radius - 1,
                       colors[jcolors[pa_id]], -1)
            if child not in ecolors.keys():
                print('bad')
                import ipdb
                ipdb.set_trace()
            cv2.line(image, (point[0], point[1]), (point_pa[0], point_pa[1]),
                     colors[ecolors[child]], radius - 2)

    # Convert back in original dtype
    if input_is_float:
        if max_val <= 1.:
            image = image.astype(np.float32) / 255.
        else:
            image = image.astype(np.float32)

    return image


J24_TO_J14 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]


def plot_pose3D(poses, x_poses=list(), inplace=True):
    """Plot the 3D pose showing the joint connections.
    J14_names = ['Right Ankle',  # 0
                 'Right Knee',  # 1
                 'Right Hip',  # 2
                 'Left Hip',  # 3
                 'Left Knee',  # 4
                 'Left Ankle',  # 5
                 'Right Wrist', # 6
                 'Right Elbow',  # 7
                 'Right Shoulder',  # 8
                 'Left Shoulder',  # 9
                 'Left Elbow',  # 10
                 'Left Wrist',  # 11
                 'Neck (LSP definition)',  # 12
                 'Head (Human3.6M definition)'  # 13
                 ]
    """

    import mpl_toolkits.mplot3d.axes3d as p3

    _CONNECTION = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [2, 8], [8, 7], [7, 6], [3, 9], [9, 10], [10, 11], [9, 12],
                   [8, 12], [12, 13]]

    def joint_color(j):
        colors = [(0, 0, 0), (255, 0, 255), (0, 0, 255),
                  (0, 255, 255), (255, 0, 0), (0, 255, 0)]
        _c = 0
        if j in [3, 4, 5]:
            _c = 1
        if j in [6, 7, 8]:
            _c = 2
        if j in [9, 10, 11]:
            _c = 3
        if j in [12, 13]:
            _c = 4
        # if j in [15, 17]:
        #     _c = 5
        return colors[_c]

    fig = plt.figure()
    if inplace:
        ax = fig.gca(projection='3d')
        # ax.view_init ( 90, 90 )  # To adjust wired world coordinate
    import math
    rows = math.ceil(math.sqrt(len(poses)))

    for i, pose in enumerate(poses):
        assert (pose.ndim == 2)
        assert (pose.shape[0] == 3) or (pose.shape[0] == 4)
        if not inplace:
            ax = fig.add_subplot(rows, rows, i + 1, projection='3d')
        for c in _CONNECTION:
            col = '#%02x%02x%02x' % joint_color(c[0])
            if pose.shape[0] == 4 and (pose[3, c[0]] == 0 or pose[3, c[1]] == 0):
                continue
            ax.plot([pose[0, c[0]], pose[0, c[1]]],
                    [pose[1, c[0]], pose[1, c[1]]],
                    [pose[2, c[0]], pose[2, c[1]]], c=col)
        for j in range(pose.shape[1]):
            col = '#%02x%02x%02x' % joint_color(j)
            if pose.shape[0] == 4 and pose[3, j] == 0:
                continue
            ax.scatter(pose[0, j], pose[1, j], pose[2, j],
                       c=col, marker='o', edgecolor=col)
        smallest = pose.min()
        largest = pose.max()
        ax.set_xlim3d(smallest, largest)
        ax.set_ylim3d(smallest, largest)
        ax.set_zlim3d(smallest, largest)
        ax.set_label(f'#{i}')
    for i, pose in enumerate(x_poses):
        assert (pose.ndim == 2)
        assert (pose.shape[0] == 3)
        if not inplace:
            ax = fig.add_subplot(rows, rows, i + 1, projection='3d')
        for c in _CONNECTION:
            col = '#%02x%02x%02x' % joint_color(c[0])
            ax.plot([pose[0, c[0]], pose[0, c[1]]],
                    [pose[1, c[0]], pose[1, c[1]]],
                    [pose[2, c[0]], pose[2, c[1]]], c=col, linestyle=':')
        for j in range(pose.shape[1]):
            col = '#%02x%02x%02x' % joint_color(j)
            ax.scatter(pose[0, j], pose[1, j], pose[2, j],
                       c=col, marker='x', edgecolor=col)
        smallest = pose.min()
        largest = pose.max()
        ax.set_xlim3d(smallest, largest)
        ax.set_ylim3d(smallest, largest)
        ax.set_zlim3d(smallest, largest)
        ax.set_label(f'#{i}')
    return fig


def plot_pose_H36M(poses, x_poses=list(), inplace=True):
    """Plot the 3D pose showing the joint connections.
    h36m_names = ['Pelvis (MPII definition)', #0
                  'Left Hip', #1
                  'Left Knee', #2
                  'Left Ankle', #3
                  'Right Hip', #4
                  'Right Knee', #5
                  'Right Ankle', #6
                  'Spine (Human3.6M definition)', #7 # To interpolate
                  'Neck (LSP definition)', #8
                  'Jaw (Human3.6M definition)', #9 # To interpolate
                  'Head (Human3.6M definition)', #10  # To interpolate
                  'Left Shoulder', #11
                  'Left Elbow', #12
                  'Left Wrist', #13
                  'Right Shoulder', #14
                  'Right Elbow', #15
                  'Right Wrist'] #16
    """

    import mpl_toolkits.mplot3d.axes3d as p3

    _CONNECTION = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [7, 11], [11, 12],
                   [12, 13], [7, 14], [14, 15], [15, 16]]

    def joint_color(j):
        colors = [(0, 0, 0), (255, 0, 255), (0, 0, 255),
                  (0, 255, 255), (255, 0, 0), (0, 255, 0)]
        _c = 0
        if j in [4, 5, 6]:
            _c = 1
        if j in [0, 7, 8, 9, 10]:
            _c = 2
        if j in [11, 12, 13]:
            _c = 3
        if j in [14, 15, 16]:
            _c = 4
        # if j in [15, 17]:
        #     _c = 5
        return colors[_c]

    fig = plt.figure()
    if inplace:
        ax = fig.gca(projection='3d')
        # ax.view_init ( 90, 90 )  # To adjust wired world coordinate
    import math
    rows = math.ceil(math.sqrt(len(poses)))

    for i, pose in enumerate(poses):
        assert (pose.ndim == 2)
        assert (pose.shape[0] == 3) or (pose.shape[0] == 4)
        if not inplace:
            ax = fig.add_subplot(rows, rows, i + 1, projection='3d')
        for c in _CONNECTION:
            col = '#%02x%02x%02x' % joint_color(c[0])
            if pose.shape[0] == 4 and (pose[3, c[0]] == 0 or pose[3, c[1]] == 0):
                continue
            ax.plot([pose[0, c[0]], pose[0, c[1]]],
                    [pose[1, c[0]], pose[1, c[1]]],
                    [pose[2, c[0]], pose[2, c[1]]], c=col)
        for j in range(pose.shape[1]):
            col = '#%02x%02x%02x' % joint_color(j)
            if pose.shape[0] == 4 and pose[3, j] == 0:
                continue
            ax.scatter(pose[0, j], pose[1, j], pose[2, j],
                       c=col, marker='o', edgecolor=col)
        smallest = pose.min()
        largest = pose.max()
        ax.set_xlim3d(smallest, largest)
        ax.set_ylim3d(smallest, largest)
        ax.set_zlim3d(smallest, largest)
        ax.set_label(f'#{i}')
    for i, pose in enumerate(x_poses):
        assert (pose.ndim == 2)
        assert (pose.shape[0] == 3)
        if not inplace:
            ax = fig.add_subplot(rows, rows, i + 1, projection='3d')
        for c in _CONNECTION:
            col = '#%02x%02x%02x' % joint_color(c[0])
            ax.plot([pose[0, c[0]], pose[0, c[1]]],
                    [pose[1, c[0]], pose[1, c[1]]],
                    [pose[2, c[0]], pose[2, c[1]]], c=col, linestyle=':')
        for j in range(pose.shape[1]):
            col = '#%02x%02x%02x' % joint_color(j)
            ax.scatter(pose[0, j], pose[1, j], pose[2, j],
                       c=col, marker='x', edgecolor=col)
        smallest = pose.min()
        largest = pose.max()
        ax.set_xlim3d(smallest, largest)
        ax.set_ylim3d(smallest, largest)
        ax.set_zlim3d(smallest, largest)
        ax.set_label(f'#{i}')
    return fig


def get_bv_verts(bboxes_t, verts_t, trans_t, img_shape, focal_length):
    R_bv = torch.zeros(3, 3)
    R_bv[0, 0] = R_bv[2, 1] = 1
    R_bv[1, 2] = -1

    bbox_area = (bboxes_t[:, 2] - bboxes_t[:, 0]) * (bboxes_t[:, 3] - bboxes_t[:, 1])
    area_mask = torch.tensor(bbox_area > bbox_area.max() * 0.05)
    verts_t, trans_t = verts_t[area_mask], trans_t[area_mask]
    verts_t = verts_t + trans_t.unsqueeze(1)
    verts_tr = torch.einsum('bij,kj->bik', verts_t, R_bv)
    verts_tfar = verts_tr  # verts_tr + trans_t.unsqueeze(1)
    p_min, p_max = verts_tfar.view(-1, 3).min(0)[0], verts_tfar.view(-1, 3).max(0)[0]
    p_center = 0.5 * (p_min + p_max)
    # trans_tr = torch.einsum('bj,kj->bk', trans_t, R_bv)
    verts_center = (verts_tfar.view(-1, 3) - p_center).view(verts_t.shape[0], -1, 3)

    dis_min, dis_max = (verts_tfar.view(-1, 3) - p_center).min(0)[0], (
            verts_tfar.view(-1, 3) - p_center).max(0)[0]
    h, w = img_shape[-2:]
    # h, w = min(h, w), min(h, w)
    ratio_max = abs(0.9 - 0.5)
    z_x = dis_max[0] * focal_length / (ratio_max * w) + torch.abs(dis_min[2])
    z_y = dis_max[1] * focal_length / (ratio_max * h) + torch.abs(dis_min[2])
    z_x_0 = (-dis_min[0]) * focal_length / (ratio_max * w) + torch.abs(
        dis_min[2])
    z_y_0 = (-dis_min[1]) * focal_length / (ratio_max * h) + torch.abs(
        dis_min[2])
    z = max(z_x, z_y, z_x_0, z_y_0)
    verts_right = verts_tfar - p_center + torch.tensor([0, 0, z])
    return verts_right

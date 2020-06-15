import mmcv
import numpy as np
import torch

__all__ = [
    'ImageTransform', 'BboxTransform', 'MaskTransform', 'SegMapTransform',
    'Numpy2Tensor', 'coco17_to_superset'
]


class ImageTransform(object):
    """Preprocess an image.

    1. rescale the image to expected size
    2. normalize the image
    3. flip the image (if needed)
    4. pad the image (if needed)
    5. transpose to (c, h, w)
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 to_rgb=True,
                 size_divisor=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.size_divisor = size_divisor

    def __call__(self, img, scale, flip=False, keep_ratio=True):
        if keep_ratio:
            if img.shape[:2] == scale:
                scale_factor = 1.
            else:
                img, scale_factor = mmcv.imrescale(img, scale, return_scale=True)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                img, scale, return_scale=True)
            scale_factor = np.array(
                [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        img_shape = img.shape
        img = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
        if flip:
            img = mmcv.imflip(img)
        if self.size_divisor is not None:
            img = mmcv.impad_to_multiple(img, self.size_divisor)
            pad_shape = img.shape
        else:
            pad_shape = img_shape
        img = img.transpose(2, 0, 1)
        return img, img_shape, pad_shape, scale_factor


def bbox_flip(bboxes, img_shape):
    """Flip bboxes horizontally.

    Args:
        bboxes(ndarray): shape (..., 4*k)
        img_shape(tuple): (height, width)
    """
    assert bboxes.shape[-1] % 4 == 0
    w = img_shape[1]
    flipped = bboxes.copy()
    flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
    flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
    return flipped


class BboxTransform(object):
    """Preprocess gt bboxes.

    1. rescale bboxes according to image size
    2. flip bboxes (if needed)
    3. pad the first dimension to `max_num_gts`
    """

    def __init__(self, max_num_gts=None):
        self.max_num_gts = max_num_gts

    def __call__(self, bboxes, img_shape, scale_factor, flip=False):
        gt_bboxes = bboxes * scale_factor
        if flip:
            gt_bboxes = bbox_flip(gt_bboxes, img_shape)
        gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1)
        gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1)
        if self.max_num_gts is None:
            return gt_bboxes
        else:
            num_gts = gt_bboxes.shape[0]
            padded_bboxes = np.zeros((self.max_num_gts, 4), dtype=np.float32)
            padded_bboxes[:num_gts, :] = gt_bboxes
            return padded_bboxes


class MaskTransform(object):
    """Preprocess masks.

    1. resize masks to expected size and stack to a single array
    2. flip the masks (if needed)
    3. pad the masks (if needed)
    """

    def __call__(self, masks, pad_shape, scale_factor, flip=False):
        masks = [
            mmcv.imrescale(mask, scale_factor, interpolation='nearest')
            for mask in masks
        ]
        if flip:
            masks = [mask[:, ::-1] for mask in masks]
        padded_masks = [
            mmcv.impad(mask, pad_shape[:2], pad_val=0) for mask in masks
        ]
        padded_masks = np.stack(padded_masks, axis=0)
        return padded_masks


class SegMapTransform(object):
    """Preprocess semantic segmentation maps.

    1. rescale the segmentation map to expected size
    3. flip the image (if needed)
    4. pad the image (if needed)
    """

    def __init__(self, size_divisor=None):
        self.size_divisor = size_divisor

    def __call__(self, img, scale, flip=False, keep_ratio=True):
        if keep_ratio:
            img = mmcv.imrescale(img, scale, interpolation='nearest')
        else:
            img = mmcv.imresize(img, scale, interpolation='nearest')
        if flip:
            img = mmcv.imflip(img)
        if self.size_divisor is not None:
            img = mmcv.impad_to_multiple(img, self.size_divisor)
        return img


class Numpy2Tensor(object):

    def __init__(self):
        pass

    def __call__(self, *args):
        if len(args) == 1:
            return torch.from_numpy(args[0])
        else:
            return tuple([torch.from_numpy(np.array(array)) for array in args])


def coco17_to_superset(coco_kpts):
    """
    kp_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder',  # 5
                    'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',  # 10
                    'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']

    0 - Right Ankle
    1 - Right Knee
    2 - Right Hip
    3 - Left Hip
    4 - Left Knee
    5 - Left Ankle
    6 - Right Wrist
    7 - Right Elbow
    8 - Right Shoulder
    9 - Left Shoulder
    10 - Left Elbow
    11 - Left Wrist
    12 - Neck (LSP definition)
    13 - Top of Head (LSP definition)
    14 - Pelvis (MPII definition)
    15 - Thorax (MPII definition)
    16 - Spine (Human3.6M definition)
    17 - Jaw (Human3.6M definition)
    18 - Head (Human3.6M definition)
    19 - Nose
    20 - Left Eye
    21 - Right Eye
    22 - Left Ear
    23 - Right Ear
    :param gt_keypoints: ...x17xM Keypoints tensor or array
    :return super_kpts
    """

    creator_fn = None
    coco_in_superset = [19, 20, 21, 22, 23, 9,  # 5
                        8, 10, 7, 11, 6,  # 10
                        3, 2, 4, 1, 5, 0  # 15
                        ]
    if isinstance(coco_kpts, torch.Tensor):
        creator_fn = torch.zeros
    elif isinstance(coco_kpts, np.ndarray):
        creator_fn = np.zeros
    super_kpts = creator_fn((coco_kpts.shape[:-2]) + (24,) + (coco_kpts.shape[-1],))
    super_kpts[..., coco_in_superset, :] = coco_kpts
    return super_kpts


def coco17to19(coco17pose):
    """
    kp_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder',  # 5
                'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',  # 10
                'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
    coco19_kp_names = ['neck', 'nose', 'hip', 'l_shoulder', 'l_elbow', 'l_wrist',  # 5
                'l_hip', 'l_knee', 'l_ankle', 'r_shoulder', 'r_elbow',  # 10
                'r_wrist', 'r_hip', 'r_knee', 'r_ankle', 'l_eye', # 15
                'l_ear', 'r_eye', 'r_ear']
    :param coco17pose: 17x3 coco pose np.array
    :return: 19x3 coco19 pose np.array
    """
    coco19pose = np.zeros((19, coco17pose.shape[1]))
    index_array = np.array([1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14])
    coco19pose[index_array] = coco17pose
    coco19pose[0] = (coco17pose[5] + coco17pose[6]) / 2
    coco19pose[2] = (coco17pose[11] + coco17pose[12]) / 2
    coco19pose[-4:] = coco17pose[0]  # Since we have not implement eye and ear yet.
    return coco19pose


def coco19_to_superset(coco19pose):
    """
    coco19_kp_names = ['neck', 'nose', 'hip', 'l_shoulder', 'l_elbow', 'l_wrist',  # 5
                'l_hip', 'l_knee', 'l_ankle', 'r_shoulder', 'r_elbow',  # 10
                'r_wrist', 'r_hip', 'r_knee', 'r_ankle', 'l_eye', # 15
                'l_ear', 'r_eye', 'r_ear']
    kpts_coco19 = [12, 19, 14, 9, 10, 11,
                    3, 4, 5, 8, 7, #10
                     6, 2, 1, 0, 20, #15
                     22, 21, 23]
    0 - Right Ankle
    1 - Right Knee
    2 - Right Hip
    3 - Left Hip
    4 - Left Knee
    5 - Left Ankle
    6 - Right Wrist
    7 - Right Elbow
    8 - Right Shoulder
    9 - Left Shoulder
    10 - Left Elbow
    11 - Left Wrist
    12 - Neck (LSP definition)
    13 - Top of Head (LSP definition)
    14 - Pelvis (MPII definition)
    15 - Thorax (MPII definition)
    16 - Spine (Human3.6M definition)
    17 - Jaw (Human3.6M definition)
    18 - Head (Human3.6M definition)
    19 - Nose
    20 - Left Eye
    21 - Right Eye
    22 - Left Ear
    23 - Right Ear
    :param coco19pose:
    :return:
    """
    pass
    # superset_names =
    J24_names = ['Right Ankle',
                 'Right Knee',
                 'Right Hip',
                 'Left Hip',
                 'Left Knee',
                 'Left Ankle',
                 'Right Wrist',
                 'Right Elbow',
                 'Right Shoulder',
                 'Left Shoulder',
                 'Left Elbow',
                 'Left Wrist',
                 'Neck (LSP definition)',
                 'Top of Head (LSP definition)',
                 'Pelvis (MPII definition)',
                 'Thorax (MPII definition)',
                 'Spine (Human3.6M definition)',
                 'Jaw (Human3.6M definition)',
                 'Head (Human3.6M definition)',
                 'Nose',
                 'Left Eye',
                 'Right Eye',
                 'Left Ear',
                 'Right Ear']
    coco19_kp_names = ['neck', 'nose', 'hip', 'l_shoulder', 'l_elbow', 'l_wrist',  # 5
                       'l_hip', 'l_knee', 'l_ankle', 'r_shoulder', 'r_elbow',  # 10
                       'r_wrist', 'r_hip', 'r_knee', 'r_ankle', 'l_eye',  # 15
                       'l_ear', 'r_eye', 'r_ear']

    h36m_names = ['Pelvis (MPII definition)',
                  'Left Hip',
                  'Left Knee',
                  'Left Ankle',
                  'Right Hip',
                  'Right Knee',
                  'Right Ankle',
                  'Spine (Human3.6M definition)',  # To interpolate
                  'Neck (LSP definition)',
                  'Jaw (Human3.6M definition)',  # To interpolate
                  'Head (Human3.6M definition)',  # To interpolate
                  'Left Shoulder',
                  'Left Elbow',
                  'Left Wrist',
                  'Right Shoulder',
                  'Right Elbow',
                  'Right Wrist']
    """
    0: Pelvis (MPII definition)
    1: Left Hip
    2: Left Knee
    3: Left Ankle
    4: Right Hip
    5: Right Knee
    6: Right Ankle
    7: Spine (Human3.6M definition)
    8: Neck (LSP definition)
    9: Jaw (Human3.6M definition)
    10: Head (Human3.6M definition)
    11: Left Shoulder
    12: Left Elbow
    13: Left Wrist
    14: Right Shoulder
    15: Right Elbow
    16: Right Wrist
    """

    superset_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]
    kpts_coco19 = [12, 19, 14, 9, 10, 11,
                   3, 4, 5, 8, 7,  # 10
                   6, 2, 1, 0, 20,  # 15
                   22, 21, 23]


def PanopticJ15_to_Superset():
    """
        0 - Right Ankle
    1 - Right Knee
    2 - Right Hip
    3 - Left Hip
    4 - Left Knee
    5 - Left Ankle
    6 - Right Wrist
    7 - Right Elbow
    8 - Right Shoulder
    9 - Left Shoulder
    10 - Left Elbow
    11 - Left Wrist
    12 - Neck (LSP definition)
    13 - Top of Head (LSP definition)
    14 - Pelvis (MPII definition)
    15 - Thorax (MPII definition)
    16 - Spine (Human3.6M definition)
    17 - Jaw (Human3.6M definition)
    18 - Head (Human3.6M definition)
    19 - Nose
    20 - Left Eye
    21 - Right Eye
    22 - Left Ear
    23 - Right Ear
    BoneJointOrder = { [2 1 3] ...   %{headtop, neck, bodyCenter}
                    , [1 4 5 6] ... %{neck, leftShoulder, leftArm, leftWrist}
                    , [3 7 8 9] ...  %{neck, leftHip, leftKnee, leftAnkle}
                    , [1 10 11 12]  ... %{neck, rightShoulder, rightArm, rightWrist}
                    , [3 13 14 15]};    %{neck, rightHip, rightKnee, rightAnkle}
    :return:
    """
    pass
    Panoptic_to_J15 = [12, 13, 14, 9, 10, 11,  # 5s
                       3, 4, 5, 8, 7, 6,  # 11
                       2, 1, 0
                       ]

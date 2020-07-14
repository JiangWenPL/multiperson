'''
Parts of the code are adapted from
https://github.com/JimmySuen/integral-human-pose/blob/master/common/utility/image_processing_cv.py
'''
import numpy as np
from tqdm import trange
import os
import cv2

def convert_to_pkl(imgname, bboxes, keypoints_2d, keypoints_3d=None, pose=None, betas=None, width=256, height=256):
    data = []
    for i in range(len(imgname)):
        datum = dict(filename=imgname[i], width=width, height=height, bboxes=bboxes[i][np.newaxis], kpts2d=keypoints_2d[i][np.newaxis])
        if keypoints_3d is not None:
            datum['kpts3d'] = keypoints_3d[i][np.newaxis]
        if pose is not None:
            datum['pose'] = pose[i][np.newaxis]
        if betas is not None:
            datum['betas'] = betas[i][np.newaxis]
        data.append(datum)
    return data

def crop_image(img, keypoints_2d, bbox, center_x, center_y, bb_width, bb_height, patch_width, patch_height):
    img_height, img_width, img_channels = img.shape

    trans, _ = gen_trans_from_patch_cv(center_x, center_y, bb_width, bb_height, patch_width, patch_height, 1.2, 0.0, inv=False)

    img = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)), flags=cv2.INTER_LINEAR)
    for n_jt in range(len(keypoints_2d)):
        keypoints_2d[n_jt, 0:2] = trans_point2d(keypoints_2d[n_jt, 0:2], trans)
    for i in range(2):
        bbox[i, :] = trans_point2d(bbox[i, :], trans)

    return img, keypoints_2d, bbox

def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y # np.array([c_x, c_y], dtype=np.float32)
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    trans_inv = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans, trans_inv



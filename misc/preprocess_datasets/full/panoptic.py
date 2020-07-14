import ipdb
import sys
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
from glob import glob
import cv2
import mmcv
import pathlib
import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser(description='Preprocess Panoptic')
parser.add_argument('panoptic_path')
parser.add_argument('dataset_name')
parser.add_argument('sequence_idx')


def projectPoints(X, K, R, t, Kd):
    """ Projects points X (3xN) using camera intrinsics K (3x3),
    extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].

    Roughly, x = K*(R*X + t) + distortion

    See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    or cv2.projectPoints
    """

    x = np.asarray(R @ X + t)

    x[0:2, :] = x[0:2, :] / x[2, :]

    r = x[0, :] * x[0, :] + x[1, :] * x[1, :]

    x[0, :] = x[0, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r) + 2 * Kd[2] * x[0, :] * x[1, :] + Kd[3] * (
            r + 2 * x[0, :] * x[0, :])
    x[1, :] = x[1, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r) + 2 * Kd[3] * x[0, :] * x[1, :] + Kd[2] * (
            r + 2 * x[1, :] * x[1, :])

    x[0, :] = K[0, 0] * x[0, :] + K[0, 1] * x[1, :] + K[0, 2]
    x[1, :] = K[1, 0] * x[0, :] + K[1, 1] * x[1, :] + K[1, 2]

    return x


body_edges = np.array(
    [[1, 2], [1, 4], [4, 5], [5, 6], [1, 3], [3, 7], [7, 8], [8, 9], [3, 13], [13, 14], [14, 15], [1, 10], [10, 11],
     [11, 12]]) - 1

colors = plt.cm.hsv(np.linspace(0, 1, 10)).tolist()
superset_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

EXPAND_FACTOR = 0.1

scale = (832, 512)
kpts_coco19 = [12, 19, 14, 9, 10, 11,
               3, 4, 5, 8, 7,  # 10
               6, 2, 1, 0, 20,  # 15
               22, 21, 23]
J24_to_J15 = [12, 13, 14, 9, 10, 11,  # 5s
              3, 4, 5, 8, 7, 6,  # 11
              2, 1, 0
              ]

def process_panoptic(panoptic_path, dataset_name, sequence_idx):
    seq_idxs = np.loadtxt(sequence_idx).astype(int)
    annotation_dir = osp.join(panoptic_path, 'processed', 'annotations')
    pathlib.Path(annotation_dir).mkdir(parents=True, exist_ok=True)

    img_dirs = glob(osp.join(panoptic_path, dataset_name, 'hdImgs', '*'))
    info_list = list()
    for img_dir in img_dirs:
        cam_name = osp.basename(img_dir)

        calib = None
        calib_file = f'{panoptic_path}/{dataset_name}/calibration_{dataset_name}.json'
        with open(calib_file) as f:
            calib = json.load(f)

        cam = None  # type: dict
        for cam in calib['cameras']:
            if cam['name'] == cam_name:
                break
        K, R, t = np.array(cam['K']), np.array(cam['R']), np.array(cam['t'])
        resized_dir = osp.join(panoptic_path, 'processed', dataset_name,
                               '{0:02d}_{1:02d}'.format(cam['panel'], cam['node']))
        pathlib.Path(resized_dir).mkdir(parents=True, exist_ok=True)

        hd_img_path = osp.join(panoptic_path, dataset_name, 'hdImgs/')
        hd_skel_json_path = osp.join(panoptic_path, dataset_name, 'hdPose3d_stage1_coco19/')
        for hd_idx in tqdm(seq_idxs):
            image_path = hd_img_path + '{0:02d}_{1:02d}/{0:02d}_{1:02d}_{2:08d}.png'.format(cam['panel'], cam['node'],
                                                                                            hd_idx)
            skel_json_fname = hd_skel_json_path + 'body3DScene_{0:08d}.json'.format(hd_idx)

            img = cv2.imread(image_path)

            re_img, scale_factor = mmcv.imrescale(img, scale, return_scale=True)
            padded_img = np.zeros((scale[1], scale[0], 3), dtype=re_img.dtype)
            padded_img[:re_img.shape[0], :re_img.shape[1]] = re_img
            h, w = re_img.shape[:2]

            with open(skel_json_fname) as dfile:
                bframe = json.load(dfile)
            bboxes = list()
            gt_kpts3d = list()
            gt_kpts2d = list()
            for body in bframe['bodies']:
                skel = np.array(body['joints19']).reshape((-1, 4)).transpose()
                pt = projectPoints(skel[0:3, :],
                                   np.array(cam['K']), cam['R'], cam['t'],
                                   cam['distCoef'])
                kpts3d_J24 = np.zeros((24, 4))
                kpts3d_J24[J24_to_J15] = skel.T
                kpts3d_J24[:, -1] = kpts3d_J24[:, -1] > 0.1
                valid = skel[3, :] > 0.1
                pt_J24 = projectPoints(kpts3d_J24.T[0:3, :], np.array(cam['K']), cam['R'], cam['t'], cam['distCoef'])
                kpts2d_J24 = np.zeros((24, 3))
                kpts2d_J24[:, :2] = pt_J24[:2].T
                kpts2d_J24[:, -1] = kpts3d_J24[:, -1]
                s_kpts2d = np.zeros_like(kpts2d_J24)
                s_kpts2d[..., -1] = kpts2d_J24[..., -1]
                s_kpts2d[..., :-1] = kpts2d_J24[..., :-1] * scale_factor
                kpts3d_cam = np.zeros_like(kpts3d_J24)
                kpts3d_cam[:, :3] = (R @ kpts3d_J24.T[:3] + t).T / 100  # cm to m
                kpts3d_cam[:, 3] = kpts3d_J24[:, 3]

                valid_kps = s_kpts2d[s_kpts2d[:, 2] > 0, :2]
                if len(valid_kps) == 0:
                    tqdm.write(f"An invalid 2D pose found in {osp.basename(image_path)}")
                    continue
                x_min, y_min = valid_kps.min(axis=0)
                x_max, y_max = valid_kps.max(axis=0)

                new_x_min = min(max(x_min - (x_max - x_min) * EXPAND_FACTOR, 0), w)
                new_y_min = min(max(y_min - (y_max - y_min) * EXPAND_FACTOR, 0), h)
                new_x_max = min(max(x_max + (x_max - x_min) * EXPAND_FACTOR, 0), w)
                new_y_max = min(max(y_max + (y_max - y_min) * EXPAND_FACTOR, 0), h)
                bbox = np.array([new_x_min, new_y_min, new_x_max, new_y_max]).round().astype(int)
                bboxes.append(bbox)
                gt_kpts2d.append(s_kpts2d)
                gt_kpts3d.append(kpts3d_cam)
            dumped_img_path = osp.join(resized_dir, osp.basename(image_path))
            cv2.imwrite(dumped_img_path, padded_img)
            filename = osp.relpath(dumped_img_path, panoptic_path)
            cur_info = {'filename': filename, 'width': padded_img.shape[1], 'height': padded_img.shape[0],
                        'bboxes': np.array(bboxes),
                        'kpts2d': np.array(gt_kpts2d),
                        'kpts3d': np.array(gt_kpts3d),
                        }
            info_list.append(cur_info)
    annotation_path = osp.join(annotation_dir, f"{dataset_name}.pkl")

    with open(annotation_path, 'wb') as f:
        pickle.dump(info_list, f)


if __name__ == '__main__':
    args = parser.parse_args()
    process_panoptic(args.panoptic_path, args.dataset_name, args.sequence_idx)

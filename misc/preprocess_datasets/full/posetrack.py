import os
import sys
import argparse
import os.path as osp
import numpy as np
import torch
import pickle
from pycocotools.coco import COCO
import argparse
import sys
from glob import glob
import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Preprocess Posetrack')
parser.add_argument('dataset_path')
parser.add_argument('out_path')

def posetrack_extract(dataset_path, out_path):
    # convert joints to global order
    joints_idx = [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]

    images_dir = dataset_path
    json_dir = osp.join(dataset_path, 'annotations', 'train')
    json_files = glob(osp.join(json_dir, '*.json'))
    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    data = []
    for fn in tqdm(json_files):
        coco = COCO(fn)
        img_ids = coco.getImgIds()
        imgs = coco.loadImgs(img_ids)
        posetrack_images = [img for img in imgs if img['is_labeled']]
        for selected_im in posetrack_images:
            datum = dict()
            filename = selected_im['file_name']
            h, w = cv2.imread(osp.join(images_dir, filename)).shape[:2]

            ann_ids = coco.getAnnIds(imgIds=selected_im['id'])
            anns = coco.loadAnns(ann_ids)
            kpts2d = list()
            bbox = list()

            for ann in anns:
                if 'bbox' in ann:
                    np_kpts2d = np.array(ann['keypoints']).reshape(-1, 3)
                    np_kpts2d[np_kpts2d[:, -1] > 0, -1] = 1
                    if np.any((np_kpts2d[np_kpts2d[..., -1] > 0] < -100) | (np_kpts2d[np_kpts2d[..., -1] > 0] > 1e4)):
                        continue
                    annbbox = np.array([ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2],
                                        ann['bbox'][1] + ann['bbox'][3]])
                    if np.any((annbbox < -100) | (annbbox > 1e4)):
                        continue
                    kpts2d.append(np_kpts2d)
                    bbox.append(annbbox)
            if not kpts2d:
                tqdm.write(str(f'No annotations in {filename} {fn}'))
                continue

            kpts2d_ = np.stack(kpts2d, axis=0)
            kpts2d = np.zeros((kpts2d_.shape[0], 24, 3))
            kpts2d[:, joints_idx] = kpts2d_

            bboxes = np.stack(bbox).astype(np.float32)
            num_persons = kpts2d.shape[0]
            datum = {'filename': selected_im['file_name'], 'width': w, 'height': h,
                     'bboxes': bboxes,
                     'labels': np.ones(bboxes.shape[0]).astype(np.int),
                     'kpts2d': kpts2d
                    }
            data.append(datum)
    out_file = os.path.join(out_path, 'train.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    args = parser.parse_args()
    posetrack_extract(args.dataset_path, args.out_path)

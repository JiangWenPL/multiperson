import os
from os.path import join
import h5py
import numpy as np
import pickle
import argparse
import cv2
from tqdm import trange
from utils import crop_image, convert_to_pkl

parser = argparse.ArgumentParser(description='Preprocess MPII')
parser.add_argument('dataset_path')
parser.add_argument('out_path')

def mpii_extract(dataset_path, out_path, out_size=256):

    # convert joints to global order
    joints_idx = [0, 1, 2, 3, 4, 5, 14, 15, 12, 13, 6, 7, 8, 9, 10, 11]

    # annotation files
    annot_file = os.path.join(dataset_path, 'annot', 'train.h5')

    # read annotations
    f = h5py.File(annot_file, 'r')
    centers, imgnames, parts, scales = \
        np.array(f['center']), f['imgname'][:], np.array(f['part']), np.array(f['scale'])
    imgnames_cropped = np.array([os.path.join('images_pretrain', str(i) + '_' + imgnames[i].decode('utf-8')) for i in range(len(imgnames))])
    imgnames = np.array([os.path.join('images', imgname.decode('utf-8')) for imgname in imgnames])
    bboxes = np.concatenate((centers - scales[:, np.newaxis] * 100, centers + scales[:, np.newaxis] * 100), axis=-1)
    visible = parts.sum(axis=-1) > 0
    kpts2d = np.zeros((parts.shape[0], 24, 3))
    kpts2d[:, joints_idx, :-1] = parts
    kpts2d[:, joints_idx, -1] = visible

    img_dir = os.path.join(dataset_path, 'images_pretrain')
    os.makedirs(img_dir, exist_ok=True)

    for i in trange(len(imgnames)):
        img = cv2.imread(os.path.join(dataset_path, imgnames[i]))
        center = centers[i]
        bb_width = scales[i] * 200
        bb_height = scales[i] * 200
        bbox = bboxes[i].copy().reshape(2, 2)
        img, keypoints_2d, bbox = crop_image(img, kpts2d[i], bbox, center[0], center[1], bb_width, bb_height, out_size, out_size)
        kpts2d[i] = keypoints_2d
        bboxes[i] = bbox.reshape(-1)
        cv2.imwrite(os.path.join(dataset_path, imgnames_cropped[i]), img)

    data = convert_to_pkl(imgnames_cropped, bboxes, kpts2d)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'train.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    args = parser.parse_args()
    mpii_extract(args.dataset_path, args.out_path)

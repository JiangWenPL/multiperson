import os
from os.path import join
import h5py
import numpy as np
import argparse
from utils import gather_per_image
import pickle

parser = argparse.ArgumentParser(description='Preprocess MPII')
parser.add_argument('dataset_path')
parser.add_argument('out_path')

def mpii_extract(dataset_path, out_path):

    # convert joints to global order
    joints_idx = [0, 1, 2, 3, 4, 5, 14, 15, 12, 13, 6, 7, 8, 9, 10, 11]

    # structs we use
    imgnames_, bboxes_, kpts2d_ = [], [], []

    # annotation files
    annot_file = os.path.join(dataset_path, 'annot', 'train.h5')

    # read annotations
    f = h5py.File(annot_file, 'r')
    centers, imgnames, parts, scales = \
        np.array(f['center']), f['imgname'][:], np.array(f['part']), np.array(f['scale'])
    imgnames = np.array([os.path.join('images', imgname.decode('utf-8')) for imgname in imgnames])
    bboxes = np.concatenate((centers - scales[:, np.newaxis] * 100, centers + scales[:, np.newaxis] * 100), axis=-1)
    visible = parts.sum(axis=-1) > 0
    kpts2d = np.zeros((parts.shape[0], 24, 3))
    kpts2d[:, joints_idx, :-1] = parts
    kpts2d[:, joints_idx, -1] = visible

    data = gather_per_image(dict(filename=imgnames, bboxes=bboxes, kpts2d=kpts2d), img_dir=dataset_path)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'train.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    args = parser.parse_args()
    mpii_extract(args.dataset_path, args.out_path)

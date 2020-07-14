import os
from os.path import join
import argparse
import numpy as np
import scipy.io as sio
import cv2
import pickle
import glob
from tqdm import trange
from utils import crop_image, convert_to_pkl

parser = argparse.ArgumentParser(description='Preprocess LSP')
parser.add_argument('dataset_path')
parser.add_argument('out_path')

def lspet_extract(dataset_path, out_path, out_size=256):

    # bbox expansion factor
    scaleFactor = 1.0

    png_path = os.path.join(dataset_path, '*.png')
    imgs = glob.glob(png_path)
    imgs.sort()

    # annotation files
    annot_file = os.path.join(dataset_path, 'joints.mat')
    joints = sio.loadmat(annot_file)['joints'].astype(np.float)

    kpts2d = np.zeros((len(imgs), 24, 3))
    bboxes = np.zeros((len(imgs), 4))
    imgnames_cropped = []

    # go over all the images
    for i in trange(len(imgs)):
        # image name
        imgname = imgs[i].split('/')[-1]
        imgname_cropped = join('images_pretrain', str(i) + '_' + imgname)
        # read keypoints
        part14 = joints[:, :2, i]
        # scale and center
        bbox = [min(part14[:,0]), min(part14[:,1]),
                max(part14[:,0]), max(part14[:,1])]
        center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
        scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200
        # update keypoints
        part = np.zeros([24,3])
        part[:14] = np.hstack([part14, np.ones([14,1])])

        img = cv2.imread(os.path.join(dataset_path, imgname))
        bb_width = scale
        bb_height = scale
        bbox = np.array(bbox).reshape(2, 2)
        img, keypoints_2d, bbox = crop_image(img, part, bbox, center[0], center[1], bb_width, bb_height, out_size, out_size)
        kpts2d[i] = keypoints_2d
        bboxes[i] = bbox.reshape(-1)
        imgnames_cropped.append(imgname_cropped)
        cv2.imwrite(os.path.join(dataset_path, imgname_cropped), img)

    data = convert_to_pkl(imgnames_cropped, bboxes, kpts2d)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'train.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    args = parser.parse_args()
    lspet_extract(args.dataset_path, args.out_path)

import os
import argparse
import sys
import cv2
import glob
import h5py
import json
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from tqdm import tqdm
import pickle
from utils import crop_image

parser = argparse.ArgumentParser(description='Preprocess MPI-INF-3DHP')
parser.add_argument('dataset_path')
parser.add_argument('out_path')

def read_calibration(calib_file, vid_list):
    Ks, Rs, Ts = [], [], []
    file = open(calib_file, 'r')
    content = file.readlines()
    for vid_i in vid_list:
        K = np.array([float(s) for s in content[vid_i*7+5][11:-2].split()])
        K = np.reshape(K, (4, 4))
        RT = np.array([float(s) for s in content[vid_i*7+6][11:-2].split()])
        RT = np.reshape(RT, (4, 4))
        R = RT[:3,:3]
        T = RT[:3,3]/1000
        Ks.append(K)
        Rs.append(R)
        Ts.append(T)
    return Ks, Rs, Ts

def train_data(dataset_path, out_path, joints_idx, scaleFactor, extract_img=False, out_size=256):

    joints17_idx = [4, 18, 19, 20, 23, 24, 25, 3, 5, 6, 7, 9, 10, 11, 14, 15, 16]

    h, w = 2048, 2048

    # training data
    user_list = range(1,9)
    seq_list = range(1,3)
    vid_list = list(range(3)) + list(range(4,9))

    counter = 0

    data = []

    img_dir = os.path.join(dataset_path, 'pretrain')
    os.makedirs(img_dir, exist_ok=True)

    for user_i in tqdm(user_list):
        for seq_i in tqdm(seq_list):
            seq_path = os.path.join(dataset_path,
                                    'S' + str(user_i),
                                    'Seq' + str(seq_i))
            # mat file with annotations
            annot_file = os.path.join(seq_path, 'annot.mat')
            annot2 = sio.loadmat(annot_file)['annot2']
            annot3 = sio.loadmat(annot_file)['annot3']
            # calibration file and camera parameters
            calib_file = os.path.join(seq_path, 'camera.calibration')
            Ks, Rs, Ts = read_calibration(calib_file, vid_list)

            for j, vid_i in enumerate(vid_list):

                # image folder
                imgs_path = os.path.join(seq_path,
                                         'imageFrames',
                                         'video_' + str(vid_i))

                # extract frames from video file
                if extract_img:

                    # if doesn't exist
                    if not os.path.isdir(imgs_path):
                        os.makedirs(imgs_path)

                    # video file
                    vid_file = os.path.join(seq_path,
                                            'imageSequence',
                                            'video_' + str(vid_i) + '.avi')
                    vidcap = cv2.VideoCapture(vid_file)

                    # process video
                    frame = 0
                    while 1:
                        # extract all frames
                        success, image = vidcap.read()
                        if not success:
                            break
                        frame += 1
                        # image name
                        imgname = os.path.join(imgs_path,
                            'frame_%06d.jpg' % frame)
                        # save image
                        cv2.imwrite(imgname, image)

                # per frame
                cam_aa = cv2.Rodrigues(Rs[j])[0].T[0]
                pattern = os.path.join(imgs_path, '*.jpg')
                img_list = sorted(glob.glob(pattern))
                for i, img_i in enumerate(img_list):

                    # for each image we store the relevant annotations
                    img_name = img_i.split('/')[-1]
                    img_name = os.path.join('S' + str(user_i),
                                            'Seq' + str(seq_i),
                                            'imageFrames',
                                            'video_' + str(vid_i),
                                            img_name)
                    joints = np.reshape(annot2[vid_i][0][i], (28, 2))[joints17_idx]
                    S17 = np.reshape(annot3[vid_i][0][i], (28, 3))/1000
                    S17 = S17[joints17_idx] - S17[4] # 4 is the root
                    bbox = [min(joints[:,0]), min(joints[:,1]),
                            max(joints[:,0]), max(joints[:,1])]

                    # check that all joints are visible
                    x_in = np.logical_and(joints[:, 0] < w, joints[:, 0] >= 0)
                    y_in = np.logical_and(joints[:, 1] < h, joints[:, 1] >= 0)
                    ok_pts = np.logical_and(x_in, y_in)
                    if np.sum(ok_pts) < len(joints_idx):
                        continue

                    part = np.zeros([24,3])
                    part[joints_idx] = np.hstack([joints, np.ones([17,1])])

                    S = np.zeros([24,4])
                    S[joints_idx] = np.hstack([S17, np.ones([17,1])])

                    # because of the dataset size, we only keep every 10th frame
                    counter += 1
                    if counter % 10 != 1:
                        continue

                    bbox = np.array(bbox).reshape(2, 2)
                    center = 0.5 * (bbox[0] + bbox[1])
                    bb_width = bbox[1, 0] - bbox[0, 0]
                    bb_height = bbox[1, 1] - bbox[0, 1]
                    box_size = max(bb_width, bb_height)
                    img_name_cropped = os.path.join('pretrain', img_name.replace('/', '_'))
                    # store the data
                    img = cv2.imread(os.path.join(dataset_path, img_name))
                    img, keypoints_2d, bbox = crop_image(img, part, bbox, center[0], center[1], box_size, box_size, out_size, out_size)
                    datum = dict(filename=img_name_cropped, width=out_size, height=out_size, bboxes=np.array(bbox)[np.newaxis], kpts2d=keypoints_2d[np.newaxis], kpts3d=S[np.newaxis])
                    datum = {}
                    data.append(datum)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'train.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)

def mpi_inf_3dhp_extract(dataset_path, out_path, extract_img=True):

    scaleFactor = 1.0
    joints_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

    train_data(dataset_path, out_path,
               joints_idx, scaleFactor,
               extract_img=extract_img)

if __name__ == '__main__':
    args = parser.parse_args()
    mpi_inf_3dhp_extract(args.dataset_path, args.out_path)

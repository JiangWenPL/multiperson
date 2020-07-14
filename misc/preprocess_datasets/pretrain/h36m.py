import os
import sys
import cv2
import glob
import h5py
import numpy as np
import argparse
from spacepy import pycdf
from utils import crop_image, convert_to_pkl

parser = argparse.ArgumentParser(description='Preprocess Human3.6M')
parser.add_argument('dataset_path')
parser.add_argument('out_path')

# Illustrative script for training data extraction
# No SMPL parameters will be included in the .npz file.
def h36m_extract(dataset_path, out_path, out_size=256):

    # convert joints to global order
    h36m_idx = [11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]
    global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

    # structs we use
    imgnames_, widths_, heights_, centers_, parts_, Ss_  = [], [], [], [], [], []

    user_list = [1, 5, 6, 7, 8]
    out_file = os.path.join(out_path, 'train.pkl')
    protocol = 1

    img_dir = os.path.join(dataset_path, 'pretrain')
    os.makedirs(img_dir, exist_ok=True)

    annotations = []
    # go over each user
    for user_i in user_list:
        user_name = 'S%d' % user_i
        # path with GT bounding boxes
        bbox_path = os.path.join(dataset_path, user_name, 'MySegmentsMat', 'ground_truth_bb')
        # path with GT 3D pose
        pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D3_Positions_mono')
        # path with GT 2D pose
        pose2d_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D2_Positions')
        # path with videos
        vid_path = os.path.join(dataset_path, user_name, 'Videos')

        # go over all the sequences of each user
        seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
        seq_list.sort()
        for seq_i in seq_list:

            # sequence info
            seq_name = seq_i.split('/')[-1]
            action, camera, _ = seq_name.split('.')
            action = action.replace(' ', '_')
            # irrelevant sequences
            if action == '_ALL':
                continue

            # 3D pose file
            poses_3d = pycdf.CDF(seq_i)['Pose'][0]

            # 2D pose file
            pose2d_file = os.path.join(pose2d_path, seq_name)
            poses_2d = pycdf.CDF(pose2d_file)['Pose'][0]

            # bbox file
            bbox_file = os.path.join(bbox_path, seq_name.replace('cdf', 'mat'))
            bbox_h5py = h5py.File(bbox_file)

            # video file
            vid_file = os.path.join(vid_path, seq_name.replace('cdf', 'mp4'))
            imgs_path = os.path.join(dataset_path, 'images')
            vidcap = cv2.VideoCapture(vid_file)

            # go over each frame of the sequence
            for frame_i in range(poses_3d.shape[0]):
                # read video frame
                success, image = vidcap.read()
                if not success:
                    break

                # check if you can keep this frame
                if frame_i % 5 == 0 and (protocol == 1 or camera == '60457274'):
                    # image name
                    imgname = '%s_%s.%s_%06d.jpg' % (user_name, action, camera, frame_i+1)

                    # save image
                    img_out = os.path.join(imgs_path, imgname)
                    h, w, _ = img_out.shape
                    cv2.imwrite(img_out, image)

                    # read GT bounding box
                    mask = bbox_h5py[bbox_h5py['Masks'][frame_i,0]].value.T
                    ys, xs = np.where(mask==1)
                    bbox = np.array([np.min(xs), np.min(ys), np.max(xs)+1, np.max(ys)+1])

                    # read GT 3D pose
                    partall = np.reshape(poses_2d[frame_i,:], [-1,2])
                    part17 = partalll[h36m_idx]
                    part = np.zeros([24,3])
                    part[global_idx, :2] = part17
                    part[global_idx, 2] = 1

                    # read GT 3D pose
                    Sall = np.reshape(poses_3d[frame_i,:], [-1,3])/1000.
                    S17 = Sall[h36m_idx]
                    S17 -= S17[0] # root-centered
                    S24 = np.zeros([24,4])
                    S24[global_idx, :3] = S17
                    S24[global_idx, 3] = 1

                    bbox = np.array(bbox).reshape(2, 2)
                    center = 0.5 * (bbox[0] + bbox[1])
                    bb_width = bbox[1, 0] - bbox[0, 0]
                    bb_height = bbox[1, 1] - bbox[0, 1]
                    box_size = max(bb_width, bb_height)
                    # store the data
                    img = cv2.imread(os.path.join(dataset_path, 'images', imgname))
                    img, keypoints_2d, bbox = crop_image(img, part, bbox, center[0], center[1], box_size, box_size, out_size, out_size)

                    datum = dict(filename=os.path.join('pretrain', imgname),
                                 width=w,
                                 height=h,
                                 bboxes=np.array(bbox)[np.newaxis],
                                 kpts2d=part[np.newaxis],
                                 kpts3d=S24[np.newaxis])
                    annotations.append(datum)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    args = parser.parse_args()
    h36m_extract(args.dataset_path, args.out_path)

import os
from os.path import join
import h5py
import numpy as np
import pickle
import argparse
import cv2
from tqdm import tqdm, trange
import json
from utils import crop_image, convert_to_pkl

parser = argparse.ArgumentParser(description='Preprocess COCO')
parser.add_argument('dataset_path')
parser.add_argument('out_path')

def coco_extract(dataset_path, out_path, version='2014', out_size=256):

    # convert joints to global order
    joints_idx = [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]

    # structs we need
    imgnames, bboxes, kpts2d = [], [], []

    # json annotation file
    json_path = os.path.join(dataset_path,
                             'annotations',
                             f'person_keypoints_train{version}.json')
    json_data = json.load(open(json_path, 'r'))

    imgs = {}
    for img in json_data['images']:
        imgs[img['id']] = img

    for annot in tqdm(json_data['annotations']):
        # keypoints processing
        keypoints = annot['keypoints']
        keypoints = np.reshape(keypoints, (17,3))
        keypoints[keypoints[:,2]>0,2] = 1
        # image name
        image_id = annot['image_id']
        img_name = str(imgs[image_id]['file_name'])
        img_name_full = join(f'train{version}', img_name)
        # keypoints
        part = np.zeros([24,3])
        part[joints_idx] = keypoints
        # scale and center
        bbox = annot['bbox']

        # store data
        imgnames.append(img_name_full)
        bboxes.append(bbox)
        kpts2d.append(part)
    imgnames = np.array(imgnames)
    img_dir = os.path.join(dataset_path, 'pretrain')
    os.makedirs(img_dir, exist_ok=True)

    imgnames_cropped = np.array([os.path.join('pretrain', str(i) + '_' + imgnames[i]) for i in range(len(imgnames))])
    bboxes = np.array(bboxes)
    kpts2d = np.array(kpts2d)
    for i in trange(len(imgnames)):
        img = cv2.imread(os.path.join(dataset_path, imgnames[i]))
        bbox = bboxes[i].reshape(2, 2)
        center = bbox[0] + 0.5*bbox[1]
        bb_width = bbox[1, 0]
        bb_height = bbox[1, 1]
        bbox[1] += bbox[0]
        box_size = max(bb_width, bb_height)
        img, keypoints_2d, bbox = crop_image(img, kpts2d[i], bbox, center[0], center[1], box_size, box_size, out_size, out_size)
        kpts2d[i] = keypoints_2d
        bboxes[i] = bbox.reshape(-1)
        cv2.imwrite(os.path.join(dataset_path, imgnames_cropped[i]), img)

    data = convert_to_pkl(imgnames_cropped, bboxes, kpts2d)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, f'train.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)
if __name__ == '__main__':
    args = parser.parse_args()
    coco_extract(args.dataset_path, args.out_path)

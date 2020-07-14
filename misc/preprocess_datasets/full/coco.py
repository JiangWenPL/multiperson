import os
from os.path import join
import sys
import json
import numpy as np
from tqdm import tqdm
import argparse
from utils import gather_per_image

parser = argparse.ArgumentParser(description='Preprocess COCO')
parser.add_argument('dataset_path')
parser.add_argument('out_path')

def coco_extract(dataset_path, out_path, version='2014'):

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
    bboxes = np.array(bboxes)
    kpts2d = np.array(kpts2d)
    data = gather_per_image(dict(filename=imgnames, bboxes=bboxes, kpts2d=kpts2d), img_dir=dataset_path)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'train.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)
if __name__ == '__main__':
    args = parser.parse_args()
    coco_extract(args.dataset_path, args.out_path)

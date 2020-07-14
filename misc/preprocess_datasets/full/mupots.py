import numpy as np
import cv2
import pickle
from glob import glob
from tqdm import tqdm
import argparse
import os.path as osp
import os

parser = argparse.ArgumentParser(description='Preprocess MUPOTS')
parser.add_argument('dataset_path')
parser.add_argument('dump_file')

def mupots_extract(dataset_path, dump_file):
    ''' Evaluation done in MATLAB so the generated pickle file has only the images '''
    TS_dirs = sorted(glob(f'{dataset_path}/TS*'))
    data = list()
    for ts in TS_dirs:
        imgs = sorted(glob(osp.join(ts, '*.jpg')))
        for image_path in tqdm(imgs):
            img = cv2.imread(image_path)
            h, w = img.shape[:2]
            data.append({'filename': image_path, 'width': w, 'height': h})

    with open(dump_file, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    args = parser.parse_args()
    mupots_extract(args.dataset_path, args.dump_file)

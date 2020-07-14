import numpy as np
from tqdm import trange
import os
import cv2
from PIL import Image
import argparse

def gather_per_image(x, img_dir):
    filenames = x['filename']
    indices = np.argsort(filenames)
    x = {k: v[indices] for k,v in x.items()}
    filenames = x['filename']

    image_boxes = [[]]
    old_name = str(filenames[0])
    img_count = 0
    for i in range(len(filenames)):
        name = str(filenames[i])
        if name != old_name:
            img_count += 1
            image_boxes.append([])
        old_name = name
        image_boxes[img_count].append(i)

    data = [{} for _ in range(len(image_boxes))]
    img_shapes = []
    for i in trange(len(image_boxes)):
        for key in x.keys():
            data[i][key] = x[key][image_boxes[i]]
        data[i]['filename'] = data[i]['filename'][0]
        img = Image.open(os.path.join(img_dir, data[i]['filename']))
        data[i]['height'] = img.size[1]
        data[i]['width'] = img.size[0]
        data[i]['bboxes'][:, :2] = np.minimum(data[i]['bboxes'][:, :2], np.zeros_like(data[i]['bboxes'][:, :2]))
        data[i]['bboxes'][:, 2] = np.minimum(data[i]['bboxes'][:, 2], img.size[0] * np.ones_like(data[i]['bboxes'][:, 2]))
        data[i]['bboxes'][:, 3] = np.minimum(data[i]['bboxes'][:, 3], img.size[1] * np.ones_like(data[i]['bboxes'][:, 3]))
    return data

# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp

import json

from collections import namedtuple

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
import pickle


from utils import smpl_to_openpose

Keypoints = namedtuple('Keypoints',
                       ['keypoints', 'gender_gt', 'gender_pd'])

Keypoints.__new__.__defaults__ = (None,) * len(Keypoints._fields)


def create_dataset(dataset='pkl', **kwargs):
    if dataset.lower() == 'pkl':
        return PklDataset(**kwargs)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

class PklDataset(Dataset):

    def __init__(self, dataset_file, dtype=torch.float32, **kwargs):
        super(PklDataset, self).__init__()

        self.dtype = dtype
        with open(dataset_file, 'rb') as f:
            self.data = pickle.load(f)
        self.cnt = 0

    def get_joint_weights(self):
        return torch.ones(24)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        keypoints_3d = torch.tensor(self.data[idx]['kpts3d'], dtype=self.dtype)
        return keypoints_3d

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.cnt >= len(self.data):
            raise StopIteration

        keypoints_3d = self[self.cnt]
        self.cnt += 1

        return keypoints_3d

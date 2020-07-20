from __future__ import division, print_function
import torch
import numpy as np
import dlib
import cv2
import os
import json
from glob import glob


def create_data_file(self):

        mask_files = sorted(glob(os.path.join('data', 'with_mask','*.jpg')))
        unmask_files = sorted(glob(os.path.join('data', 'without_mask', '*.jpg')))

        mask_imgs = [cv2.imread(file) for file in mask_files]
        unmask_imgs = [cv2.imread(file) for file in unmask_files]

        mask_bbox = [self.detect_face(img) for img in mask_imgs]
        unmask_bbox = [self.detect_face(img) for img in unmask_imgs]

        mask_label = [1]*len(mask_files)
        unmask_label = [0]*len(mask_files)

        # pack data
        face_data_mask = [data for data in zip(
            mask_files, mask_bbox, mask_label) if data[1][0] == 1]
        face_data_unmask = [data for data in zip(
            unmask_files, unmask_bbox, unmask_label) if data[1][0] == 1]

        # split train test data
        def _split_train_test(data, prob=0.8):
            np.random.shuffle(data)
            train_index = int(prob * len(data))
            return data[:train_index], data[train_index:]

        train_mask, test_mask = _split_train_test(face_data_mask)
        train_unmask, test_unmask = _split_train_test(face_data_unmask)

        # export data
        def _export_to_json(data, filename):
            face_data = {d[0]: {'x': int(d[1][1]),
                               'y': int(d[1][2]),
                               'w': int(d[1][3]),
                               'h': int(d[1][4]),
                               'mask': int(d[2])}
                         for idx, d in enumerate(data) if d[1][0] == 1}
            
            with open(os.path.join('data', filename), 'w') as f:
                json.dump(face_data, f)

        _export_to_json(train_mask+train_unmask, 'train_dataset.json')
        _export_to_json(test_mask+test_unmask, 'test_dataset.json')
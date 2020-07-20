from __future__ import division, print_function
import torch
import numpy as np
import dlib
import cv2
import os
import json
from glob import glob


def detect_face(image):
        '''
        Return (is_detect, x, y, width, height)
        '''

        def _detect_face_opencv(img):
            face_cascade = cv2.CascadeClassifier(
                os.path.join('artifacts', 'opencv_frontal_face.xml'))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dets = face_cascade.detectMultiScale(gray, 1.1, 4)
            return dets

        def _detect_face_dlib(img):
            detector = dlib.get_frontal_face_detector()
            dets = detector(img, 1)
            return dets

        # prioritize dlib detector
        dets = _detect_face_dlib(image)
        if list(dets) == []:  # if dlib not found face
            dets = _detect_face_opencv(image)
            for d in dets:
                # (x, y, w, h) = d[0]+ d[2]//2, d[1]+ d[3]//2, d[2], d[3] # center xy 
                (x, y, w, h) = d[0]+ d[2]//2, d[1]+ d[3]//2, d[2], d[3]
        else:
            for d in dets:
                (x, y, w, h) = (d.left(), 
                                d.top(),
                                d.right() - d.left(), 
                                d.bottom() - d.top())

        try:
            return (1, x, y, w, h)

        except:
            return (0, 0, 0, 0, 0)


def create_data_file_1():

    mask_files = sorted(glob(os.path.join('data', 'with_mask','*.jpg')))
    unmask_files = sorted(glob(os.path.join('data', 'without_mask', '*.jpg')))

    mask_imgs = [cv2.imread(file) for file in mask_files]
    unmask_imgs = [cv2.imread(file) for file in unmask_files]

    mask_bbox = [detect_face(img) for img in mask_imgs]
    unmask_bbox = [detect_face(img) for img in unmask_imgs]

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

    def _to_dict(data):
        data = {d[0]: { 'x': int(d[1][1]),
                        'y': int(d[1][2]),
                        'w': int(d[1][3]),
                        'h': int(d[1][4]),
                        'mask': int(d[2])}
                    for idx, d in enumerate(data) if d[1][0] == 1}
        return data

    train_data = _to_dict(train_mask+train_unmask)
    test_data = _to_dict(test_mask+test_unmask)

    return train_data, test_data


def create_data_file_2():

    train_image_path =  os.path.join('data', 'yolo', 'images', 'train')
    train_label_path =  os.path.join('data', 'yolo', 'labels', 'train')

    val_image_path =  os.path.join('data', 'yolo', 'images', 'valid')
    val_label_path =  os.path.join('data', 'yolo', 'labels', 'valid')

    def _get_data(image_path, label_path):

        images_filepath = sorted(glob(os.path.join(image_path,'*.jpg')))
        # labels_filepath = sorted(glob(os.path.join(train_label_path,'*.txt')))
        imgs = [cv2.imread(file) for file in images_filepath]
        
        # loop throght image file for consistancy since filename of image and label are the same
        ob_data = []
        for file in images_filepath: 
            with open(os.path.join(label_path,file[len(image_path)+1:-3]+'txt'), 'r') as f:
                data = [float(coord) for coord in f.read().replace('\n','').split(' ')] + [file] # concat image filepath
                ob_data.append(data)
        
        return ob_data, imgs

    def _to_dict(data, imgs):

        data = {d[5]: { 'x': int(d[1]*imgs[idx].shape[0]) - int(d[3]*imgs[idx].shape[0])//2,
                        'y': int(d[2]*imgs[idx].shape[1]) - int(d[4]*imgs[idx].shape[1])//2,
                        'w': int(d[3]*imgs[idx].shape[0]),
                        'h': int(d[4]*imgs[idx].shape[1]),
                        'mask': int(d[0])}
                    for idx, d in enumerate(data)}
        return data
    
    train_data, train_imgs = _get_data(train_image_path, train_label_path)
    val_data, val_imgs = _get_data(val_image_path, val_label_path)
    train_data = _to_dict(train_data, train_imgs)
    val_data = _to_dict(val_data,val_imgs)

    return train_data, val_data


def main():
    print('[INFO] Prep Data #1')
    train_1, val_1 = create_data_file_1()
    print('[INFO] Prep Data #2')
    train_2, val_2 = create_data_file_2()
    train_dict = {**train_1, **train_2}
    val_dict = {**val_1, **val_2}

    print('[INFO] Export Data')
    with open(os.path.join('data', 'train_data.json'), 'w') as f:
        json.dump(train_dict, f)

    with open(os.path.join('data', 'val_data.json'), 'w') as f:
        json.dump(val_dict, f)


if __name__ == "__main__":
    main()
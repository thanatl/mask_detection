import future
import torch
import numpy as np
import dlib
import cv2
import os
import json
from glob import glob

# In coco format, bbox = [xmin, ymin, width, height]
# In pytorch, the input should be [xmin, ymin, xmax, ymax]


class Generator:

    def __init__(self, is_train):

        self.json_file = os.path.join('data','train_dataset.json') if is_train else os.path.join('data','test_dataset.json')
        if not os.path.isfile(self.json_file):
            self.create_data_file()

        with open(self.json_file, 'r') as f:
            self.data = json.load(f)

        self.image_file = [file for file in self.data.keys()]
        np.random.shuffle(self.image_file)
         
    @staticmethod
    def read_image(file):
        image = cv2.imread(file) #(height, width, channel)
        # image = image.reshape((image.shape[2], image.shape[0], image.shape[1])) #(channel, height, width)
        return image

    def convert_coord(x, y, w, h, width=None, height=None):
        pass

    def get_data(self, data_file):
        label = self.data[data_file]
        image = self.read_image(data_file)
        return image, np.array([label['x'], label['y'], label['w'], label['h'], label['mask']]).astype('float32')

    def __getitem__(self, index):
        return self.get_data(self.image_file[index])

    def __len__(self):
        return len(self.image_file)

    @staticmethod
    def detect_face(image):
        '''
        Return (is_detect, x (center), y (center), width, height)
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
                (x, y, w, h) = d[0]+ d[2]//2, d[1]+ d[3]//2, d[2], d[3]
        else:
            for d in dets:
                (x, y, w, h) = (d.left() + (d.right() - d.left())//2, 
                                d.top() + (d.bottom() - d.top())//2,
                                d.right() - d.left(), 
                                d.bottom() - d.top())

        try:
            return (1, x, y, w, h)

        except:
            return (0, 0, 0, 0, 0)

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


def collate_fn(batch):
    '''
    Padding and resize image by maximum image size in each batch
    return tensor of image and bounding box
    '''

    max_w = 0
    max_h = 0
    for img, bbox in batch:
        max_w = img.shape[2] if img.shape[2] > max_w else max_w
        max_h = img.shape[1] if img.shape[1] > max_h else max_h
    size = max_w if max_w > max_h else max_h

    img_tensor = torch.FloatTensor(len(batch), 3, size, size)
    img_tensor.zero_()
    bbox_tensor = torch.FloatTensor(len(batch), 5)
    bbox_tensor.zero_()
    for idx, data in enumerate(batch): 
        img, bbox = data
        img, bbox = resize_img_bbox_letterbox(img, bbox, size)
        bbox = normalize_bbox(img, bbox)
        img = image_pytorch_format(img)
        img = torch.from_numpy(img)
        bbox = torch.from_numpy(bbox)
        img_tensor[idx, :, :,:] = img
        bbox_tensor[idx,:] = bbox

    return img_tensor, bbox_tensor


def normalize_bbox(img, bbox):
    '''
    Arugments:
    img - image array (channel, height, width)
    bbox - bounding box (center x, center y, width, height, mask_label)
    size - the resize image

    Return:
    bbox - normalize bounding box according to image size
    '''
    w,h = img.shape[1], img.shape[0]

    bbox[0] = bbox[0]/w
    bbox[2] = bbox[2]/w
    bbox[1] = bbox[1]/h
    bbox[3] = bbox[3]/h

    return bbox


def resize_img_bbox_letterbox(img, bbox, size):
    '''
    Arugments:
    img - image array (channel, height, width)
    bbox - bounding box (center x, center y, width, height, mask_label)
    size - the resize image

    Return:
    img - resize image as a letter box padding, keep the original aspect ratio of the image and padding the smaller aspect of the image
    bbox - resize bounding box according to the image 
    
    '''
    
    w,h = img.shape[1], img.shape[0]
    scale = min(size/w, size/h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized_image = cv2.resize(img, (new_w, new_h))
    print(resized_image.shape)
    canvas = np.full((size, size, 3), 0)
    canvas[(size-new_h)//2:(size-new_h)//2 + new_h,(size-new_w)//2:(size-new_w)//2 + new_w, :] = resized_image
    canvas = canvas.astype(np.uint8)
   
    bbox[:4] = bbox[:4] * scale
    
    # add padding h w
    bbox[:4] += np.array([(size - new_w)/2, (size - new_h)/2, 0, 0]).astype(int)
    
    return canvas, bbox

def image_pytorch_format(img):
    return img.reshape((img.shape[2], img.shape[0], img.shape[1]))

if __name__ == "__main__":

    gen = Generator(is_train=True)

    train_dataset = Generator(is_train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, 
                                               num_workers=0, collate_fn=collate_fn)

    flag = True
    for data in train_loader:
        x, y = data
        print(y)
        break
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


class MaskDataLoader:

    def __init__(self, is_train, resize=416):

        self.resize = resize
        self.json_file = os.path.join(
            'data', 'train_data.json') if is_train else os.path.join('data', 'val_data.json')
        if not os.path.isfile(self.json_file):
            raise ValueError('Json {} Not Found'.format(self.json_file))

        with open(self.json_file, 'r') as f:
            self.data = json.load(f)

        self.image_file = [file for file in self.data.keys()]
        np.random.shuffle(self.image_file)

    @staticmethod
    def read_image(file):
        image = cv2.imread(file)  # (height, width, channel)
        # image = image.reshape((image.shape[2], image.shape[0], image.shape[1])) #(channel, height, width)
        return image

    def convert_coord(x, y, w, h, width=None, height=None):
        pass

    def get_data(self, data_file):
        label = self.data[data_file]
        image = self.read_image(data_file)
        image, label = resize_img_bbox_letterbox(image, label, self.resize)
        image = image_pytorch_format(image)
        return image, np.array([label['x'], label['y'], label['w'], label['h'], label['mask']]).astype('float32')

    def __getitem__(self, index):
        return self.get_data(self.image_file[index])

    def __len__(self):
        return len(self.image_file)


def normalize_bbox(img, bbox):
    '''
    Arugments:
    img - image array (channel, height, width)
    bbox - bounding box (center x, center y, width, height, mask_label)
    size - the resize image

    Return:
    bbox - normalize bounding box according to image size
    '''
    w, h = img.shape[1], img.shape[0]

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

    w, h = img.shape[1], img.shape[0]
    scale = min(size/w, size/h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized_image = cv2.resize(img, (new_w, new_h))
    print(resized_image.shape)
    canvas = np.full((size, size, 3), 0)
    canvas[(size-new_h)//2:(size-new_h)//2 + new_h, (size-new_w) //
           2:(size-new_w)//2 + new_w, :] = resized_image
    canvas = canvas.astype(np.uint8)

    bbox[:4] = bbox[:4] * scale

    # add padding h w
    bbox[:4] += np.array([(size - new_w)/2,
                          (size - new_h)/2, 0, 0]).astype(int)

    return canvas, bbox


def image_pytorch_format(img):
    return img.reshape((img.shape[2], img.shape[0], img.shape[1]))


def to_gpu(x):

    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def batch_to_gpu(data):
    img, target = data
    x,y,w,h,mask = target
    img = to_gpu(img).float()
    x = to_gpu(x).float()
    y = to_gpu(y).float()
    w = to_gpu(w).float()
    h = to_gpu(h).float()
    mask = to_gpu(mask).float()
    return img, (x, y, w, h, mask)


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

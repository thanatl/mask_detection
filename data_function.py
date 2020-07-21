import future
import torch
import numpy as np
import dlib
import cv2
from PIL import Image
import os
import json
from glob import glob
from transforms import transform, image_pytorch_format

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = image.reshape((image.shape[2], image.shape[0], image.shape[1])) #(channel, height, width)
        return image

    @staticmethod
    def convert_to_PIL(img):
        return Image.fromarray(img)

    @staticmethod
    def extract_coord_label(data):
        data = np.array([[item['x'], item['y'], item['w'], item['h'], item['mask']] for item in data.values()]).astype('float32')
        return data[:,:4], data[:,4].reshape((-1,1))

    @staticmethod
    def convert_coord_center_xy_wh(coord):
        return np.array([[c[0]-c[2]//2, c[1]-c[3]//2, c[2], c[3]] for c in coord]).astype('float32')

    @staticmethod
    def convert_coord_min_max_xy(coord):
        return np.array([[c[0], c[1], c[0] + c[2] , c[1] + c[3]] for c in coord]).astype('float32')

    def get_data(self, data_file):
        coord, label = self.extract_coord_label(self.data[data_file])
        coord = self.convert_coord_center_xy_wh(coord)
        image = self.read_image(data_file)
        image = self.convert_to_PIL(image)
        coord = torch.FloatTensor(coord)  # (n_objects, 4)
        label = torch.LongTensor(label)
        image, coord, label = transform(image, coord, label, self.resize, is_train=True)
        return image, coord, label

    def __getitem__(self, index):
        return self.get_data(self.image_file[index])

    def __len__(self):
        return len(self.image_file)

def collate_fn(batch):
    '''
    Return image tensor, label tensor with padding bbox, number of max bbox in that image
    '''
    # Right zero-pad all one-hot text sequences to max input length
    max_input_len = sorted([x[1].size(0) for x in batch], reverse=True)[0]
    bbox_tensor = torch.FloatTensor(len(batch), max_input_len, 4)
    channel = batch[0][0].size(0)
    size = batch[0][0].size(1)
    image_tensor = torch.FloatTensor(len(batch), channel, size, size)
    bbox_tensor.zero_()
    bbox_count_tensor = torch.LongTensor(len(batch), 1)
    target_tensor = torch.LongTensor(len(batch), max_input_len, 1)
    target_tensor.zero_()
    for idx, data in enumerate(batch):
        bbox = data[1]
        label = data[2]
        bbox_count_tensor[idx] = bbox.size(0)
        bbox_tensor[idx, :bbox.size(0), :] = bbox
        target_tensor[idx, :label.size(0), :] = label
        image_tensor[idx, :,:,:] = data[0]
    return image_tensor, bbox_tensor, target_tensor, bbox_count_tensor


def to_gpu(x):

    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def batch_to_gpu(data):
    img, coord, target, count = data
    img = to_gpu(img).float()
    target = to_gpu(target).long()
    coord = to_gpu(coord).float()
    count = to_gpu(count).float()
    return img, (coord, target, count)


if __name__ == "__main__":

    # example usage:
    # note: torch.utils.data.dataloader.default_collate doenst work
    train_dataset = MaskDataLoader(is_train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True,
                                               num_workers=0, collate_fn=collate_fn)
    for data in train_loader:
        img, target = batch_to_gpu(data)
        print(target)
        break
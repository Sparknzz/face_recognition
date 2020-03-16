from __future__ import print_function, division
import torch
import os
import numpy as np

import random
import logging

from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from dataloaders import custom_transforms as tr
from PIL import Image

class FaceDataset(Dataset):

    def __init__(self, data_shape, shuffle=False, aug_list = None, mean = None,
                 rand_mirror = False, cutoff = 0, color_jittering = 0,
                 images_filter = 0, split='train'):

        super(FaceDataset, self).__init__()

        self.split = split

        self.data_shape = data_shape        
        self.shuffle = shuffle
        self.image_size = '%d,%d'%(data_shape[1],data_shape[2])
        self.rand_mirror = rand_mirror
        print('rand_mirror', rand_mirror)

        self.cutoff = cutoff
        self.color_jittering = color_jittering


        all_df = pd.read_csv('CASIA.csv')
        self.images = all_df['image'].tolist()
        self.labels = all_df['label'].tolist()

        assert (len(self.images) == len(self.labels))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __getitem__(self, index):
        _img = np.asarray(Image.open(self.images[index]).convert('RGB'))
        _label = self.labels[index]

        if self.split == "train":
            return self.transform_tr(_img), _label
        elif self.split == 'val':
            return self.transform_val(_img), _label


    def transform_tr(self, _img):

        if _img.shape[0] != self.data_shape[1]:
            _img = tr.do_short_resize(_img, self.data_shape[1])

        if self.rand_mirror:
            _rd = random.randint(0,1)
            if _rd==1:
                _img = tr.do_flip_lr(_img)

        # normalize image
        _img = _img-127.5
        _img = _img/128

        # to tensor
        _img = np.array(_img).astype(np.float32).transpose((2, 0, 1))
        _img = torch.from_numpy(_img).float()

        return _img

    def transform_val(self, _img):        
        # normalize image
        _img = _img - 127.5
        _img = _img/128

        return _img

    def __str__(self):
        num = len(self)
        pos = (self.df['label'] == 1).sum()
        neg = num - pos

        # ---
        string = ''
        string += '\tsplit   = %s\n' % self.split
        string += '\tlen       = %8d\n' % len(self)

        string += '\t\t pos, neg = %5d  %0.3f,  %5d  %0.3f\n' % (pos, pos / num, neg, neg / num)

        return string

    def __len__(self):
        return len(self.images)



if __name__ == '__main__':
    dataset = FaceDataset(root='/data/Datasets/fv/dataset_v1.1/dataset_mix_aligned_v1.1',
                      data_list_file='/data/Datasets/fv/dataset_v1.1/mix_20w.txt',
                      phase='test',
                      input_shape=(1, 128, 128))

    trainloader = data.DataLoader(dataset, batch_size=10)

    for i, (data, label) in enumerate(trainloader):
        # imgs, labels = data
        # print imgs.numpy().shape
        # print data.cpu().numpy()
        # if i == 0:
        img = torchvision.utils.make_grid(data).numpy()
        # print img.shape
        # print label.shape
        # chw -> hwc
        img = np.transpose(img, (1, 2, 0))
        # img *= np.array([0.229, 0.224, 0.225])
        # img += np.array([0.485, 0.456, 0.406])
        img += np.array([1, 1, 1])
        img *= 127.5
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]

        cv2.imshow('img', img)
        cv2.waitKey()
        # break
        # dst.decode_segmap(labels.numpy()[0], plot=True)
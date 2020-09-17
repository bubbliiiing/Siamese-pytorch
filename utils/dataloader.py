
import os
import cv2
import torch
import math
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from random import shuffle
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


class SiameseDataset(Dataset):
    def __init__(self, image_size, dataset_path, num_train, num_val, train_ratio=0.9, train=True):
        super(SiameseDataset, self).__init__()

        self.dataset_path = dataset_path
        self.image_height = image_size[0]
        self.image_width = image_size[1]
        self.channel = image_size[2]
        
        self.train_dictionary = {}
        self._train_alphabets = []
        self._validation_alphabets = []

        self._current_train_alphabet_index = 0
        self._current_val_alphabet_index = 0

        self.train_ratio = train_ratio

        self.num_train = num_train
        self.num_val = num_val

        self.train = train

        self.load_dataset()
        self.split_train_datasets()

    def __len__(self):
        if self.train:
            return self.num_train
        else:
            return self.num_val

    def load_dataset(self):
        # 遍历dataset文件夹下面的images_background文件夹
        train_path = os.path.join(self.dataset_path, 'images_background')
        for alphabet in os.listdir(train_path):
            # 然后遍历images_background下的每一个文件夹，代表一个大种类
            alphabet_path = os.path.join(train_path, alphabet)
            current_alphabet_dictionary = {}
            for character in os.listdir(alphabet_path):
                # 在大众类下遍历小种类。
                character_path = os.path.join(alphabet_path, character)
                current_alphabet_dictionary[character] = os.listdir(character_path)
            # 获得的train_dictionary有两层，一层是大种类，另一层是小种类
            self.train_dictionary[alphabet] = current_alphabet_dictionary

    def split_train_datasets(self):
        available_alphabets = list(self.train_dictionary.keys())
        number_of_alphabets = len(available_alphabets)
        # 进行验证集和训练集的划分
        self._train_alphabets = available_alphabets[:int(self.train_ratio*number_of_alphabets)]

        self._validation_alphabets = available_alphabets[int(self.train_ratio*number_of_alphabets):]

    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, flip_signal=False):
        image = image.convert("RGB")

        h, w = input_shape
        # resize image
        rand_jit1 = rand(1-jitter,1+jitter)
        rand_jit2 = rand(1-jitter,1+jitter)
        new_ar = w/h * rand_jit1/rand_jit2

        scale = rand(0.75,1.25)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        
        # flip image or not
        flip = rand()<.5
        if flip and flip_signal: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # place image
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (255,255,255))

        new_image.paste(image, (dx, dy))
        image = new_image

        rotate = rand()<.5
        if rotate: 
            angle=np.random.randint(-5,5)
            a,b=w/2,h/2
            M=cv2.getRotationMatrix2D((a,b),angle,1)
            image=cv2.warpAffine(np.array(image),M,(w,h),borderValue=[255,255,255]) 

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255
        if self.channel==1:
            image_data = Image.fromarray(np.uint8(image_data)).convert("L")
        # cv2.imshow("123",np.uint8(image_data))
        # cv2.waitKey(0)
        return image_data

    def _convert_path_list_to_images_and_labels(self, path_list):
        number_of_pairs = int(len(path_list) / 2)
        pairs_of_images = [np.zeros((number_of_pairs, self.channel, self.image_height, self.image_width)) for i in range(2)]
        labels = np.zeros((number_of_pairs, 1))

        for pair in range(number_of_pairs):
            image = Image.open(path_list[pair * 2])
            image = self.get_random_data(image, [self.image_height, self.image_width])
            image = np.asarray(image).astype(np.float64)
            # cv2.imwrite("img/"+str(pair)+"_0"+".jpg",np.uint8(image),)
            image = np.transpose(image,[2,0,1])
            image = image / 255
            if self.channel == 1:
                pairs_of_images[0][pair, 0, :, :] = image
            else:
                pairs_of_images[0][pair, :, :, :] = image

            image = Image.open(path_list[pair * 2 + 1])
            image = self.get_random_data(image, [self.image_height, self.image_width])
            image = np.asarray(image).astype(np.float64)
            # cv2.imwrite("img/"+str(pair)+"_1"+".jpg",np.uint8(image),)
            image = np.transpose(image,[2,0,1])
            image = image / 255
            if self.channel == 1:
                pairs_of_images[1][pair, 0, :, :] = image
            else:
                pairs_of_images[1][pair, :, :, :] = image

            if (pair + 1) % 2 == 0:
                labels[pair] = 0
            else:
                labels[pair] = 1

        # 随机的排列组合
        random_permutation = np.random.permutation(number_of_pairs)
        labels = labels[random_permutation]
        pairs_of_images[0][:, :, :, :] = pairs_of_images[0][random_permutation, :, :, :]
        pairs_of_images[1][:, :, :, :] = pairs_of_images[1][random_permutation, :, :, :]
        # print(path_list[number_of_pairs],labels)
        return pairs_of_images, labels

    def __getitem__(self, index):
        if self.train:
            if self._current_train_alphabet_index==0:
                shuffle(self._train_alphabets)
            current_alphabet = self._train_alphabets[self._current_train_alphabet_index]
            self._current_train_alphabet_index = (self._current_train_alphabet_index + 1) // len(self._train_alphabets)
        else:
            if self._current_val_alphabet_index==0:
                shuffle(self._validation_alphabets)
            current_alphabet = self._validation_alphabets[self._current_val_alphabet_index]
        
        # 判断大类别里面的小类别的名称
        available_characters = list(self.train_dictionary[current_alphabet].keys())
        number_of_characters = len(available_characters)

        batch_images_path = []

        # 在小类别里面筛选
        index = random.randint(0, number_of_characters-1)
        
        # 除去小类别的名称
        current_character = available_characters[index]
        # 获取当前这个小类别的路径
        image_path = os.path.join(self.dataset_path, 'images_background', current_alphabet, current_character)

        available_images = (self.train_dictionary[current_alphabet])[current_character]
        image_indexes = random.sample(range(0, len(available_images)), 3)
        # 取出两张类似的图片
        image = os.path.join(image_path, available_images[image_indexes[0]])
        batch_images_path.append(image)
        image = os.path.join(image_path, available_images[image_indexes[1]])
        batch_images_path.append(image)

        # 取出两张不类似的图片
        image = os.path.join(image_path, available_images[image_indexes[2]])
        batch_images_path.append(image)
        # 取出与当前的小类别不同的类
        different_characters = available_characters[:]
        different_characters.pop(index)
        different_character_index = random.sample(range(0, number_of_characters - 1), 1)
        current_character = different_characters[different_character_index[0]]
        image_path = os.path.join(self.dataset_path, 'images_background', current_alphabet, current_character)

        available_images = (self.train_dictionary[current_alphabet])[current_character]
        image_indexes = random.sample(range(0, len(available_images)), 1)
        image = os.path.join(image_path, available_images[image_indexes[0]])
        batch_images_path.append(image)

        images, labels = self._convert_path_list_to_images_and_labels(batch_images_path)
        return images, labels

# DataLoader中collate_fn使用
def dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.concatenate(np.array(images),axis=1)
    bboxes = np.concatenate(np.array(bboxes),axis=0)
    return images, bboxes


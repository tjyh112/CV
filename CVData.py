from torch.utils.data import Dataset
from PIL import Image
from skimage.io import imread_collection
# import cv2
import numpy as np
import torch
from utils import *
import torchvision.transforms as transforms
from PIL import ImageFile
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CVData(Dataset):

    def __init__(self, path, train, transform, train_ratio=0.8, resize_width=360, resize_height=240):

        super()
        self.path = path
        self.train = train
        self.train_ratio = train_ratio
        self.transform = transform
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.data = []
        self.targets = []
        self.folders = get_folders(self.path)
        self.load_data()


    def resize(self, path):

        collections = imread_collection(path)
        for file in collections.files:
            img = cv2.imread(file)
            if img.shape[0] == self.resize_height and img.shape[1] == self.resize_width:
                continue
            arr = cv2.resize(img, (self.resize_width, self.resize_height))
            Image.fromarray(arr).save(file)



    def load_data(self, resize=False):

        if resize:
            for folder in self.folders:
                path = folder + '/*.jpg'
                self.resize(path)


        #load image
        # car_path = self.path + '/car/*.jpg'
        # motorbike_path = self.path + '/motorbike/*.jpg'
        # background_path = self.path +'/background/*.jpg'
        # face_path = self.path + '/face/*.jpg'
        # airplane_path = self.path + '/airplane/*.jpg'
        # leaf_path = self.path + '/leaf/*.jpg'
        data = np.array([], ndmin=4).reshape((0, self.resize_height, self.resize_width, 3))
        targets = np.array([])
        for i, path in enumerate(self.folders):
            collection = imread_collection(path+'/*.jpg')
            array = np.array(collection)
            label = np.array([i for _ in range(array.shape[0])])
            data = np.append(data, array, axis=0)
            targets = np.append(targets, label, axis=0)

        # car_collection = imread_collection(car_path)
        # motorbike_collection = imread_collection(motorbike_path)
        # background_collection = imread_collection(background_path)
        # face_collection = imread_collection(face_path)
        # airplane_collection = imread_collection(airplane_path)
        # leaf_collection = imread_collection(leaf_path)
        #
        #
        #
        # car_array = np.array(car_collection)
        # car_label = np.zeros(car_array.shape[0])
        #
        # motorbike_array = np.array(motorbike_collection)
        # motorbike_label = np.ones(motorbike_array.shape[0])
        #
        # background_array = np.array(background_collection)
        # background_label = np.array([2 for x in range(background_array.shape[0])])
        #
        # face_array = np.array(face_collection)
        # face_label = np.array([3 for _ in range(face_array.shape[0])])
        #
        # airplane_array = np.array(airplane_collection)
        # airplane_label = np.array([4 for _ in range(airplane_array.shape[0])])
        #
        # leaf_array = np.array(leaf_collection)
        # leaf_label = np.array([5 for _ in range(leaf_array.shape[0])])
        #
        #
        #
        # data = np.concatenate((car_array, motorbike_array, background_array, face_array, airplane_array, leaf_array), axis=0)
        # targets = np.concatenate((car_label, motorbike_label, background_label, face_label,
        #                           airplane_label, leaf_label), axis=0)



        # train set and test set saprate
        # k = int(data.shape[0] * self.train_ratio)
        # train_index = random_int_list(0, data.shape[0]-1, k)
        #
        # if self.train:
        #     self.data = data[0:k, :]
        #     self.targets = targets[0:k]
        #
        # else:
        #     self.data = data[k:, :]
        #     self.targets = targets[k:]

        self.data = data
        self.targets = targets

    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]


        if self.transform is not None:
            img = self.transform(img)

        # img = img.transpose(0,2)
        return img, target

    def __len__(self):
        return len(self.data)



if __name__ == '__main__':
    data = CVData(path='./dataset/train', transform=None, train=True)
    Data_loader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True,
                                                      num_workers=0)

    print()


# -*- coding: utf-8 -*-


import cv2
import keras
import pydicom
import numpy as np
import albumentations as A
from mask_functions import rle2mask

class DataGenerator(keras.utils.Sequence):
    def __init__(self, file_path_list, labels, augmentations=None, batch_size=32, 
                 img_size=256, n_channels=1, shuffle=True):
        self.file_path_list = file_path_list
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augmentations
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.file_path_list)) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # get list of IDs
        file_path_list_temp = [self.file_path_list[k] for k in indexes]
        
        # Generate data
        X, y = self.__data_generation(file_path_list_temp)
          
        # return data 
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.file_path_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, file_path_list_temp):
        'Generates data containing batch_size samples'
        # Initialization self.batch_size
        X = np.empty((self.batch_size, self.img_size, self.img_size, self.n_channels))
        y = np.empty((self.batch_size, self.img_size, self.img_size, 1))
        y_class = []
        for idx, file_path in enumerate(file_path_list_temp):
            
            id = file_path.split('/')[-1][:-4]
            rle = self.labels.get(id)
            
            image = pydicom.read_file(file_path).pixel_array

            if len(image.shape)==2:
                image = np.repeat(image[...,None],3,2)
   
            image_resized = cv2.resize(image, (self.img_size, self.img_size))
            X[idx,] = image_resized
            
            # if there is no mask create empty mask
            # notice we are starting of with 1024 because we need to use the rle2mask function
            if rle is None:
                mask = np.zeros((1024, 1024))
            else:
                if len(rle) == 1:
                    mask = rle2mask(rle[0], 1024, 1024).T
                else: 
                    mask = np.zeros((1024, 1024))
                    for r in rle:
                        mask =  mask + rle2mask(r, 1024, 1024).T
            
            # Store class
            y[idx,] = cv2.resize(mask,(self.img_size,self.img_size))[..., np.newaxis]
            y[y>0] = 255
        
        if self.augment is None:
            return X / 255.0, np.array(y) / 255
        else:
            X = np.uint8(X)
            y = np.uint8(y)
            im,mask = [],[]   
            for m,k in zip(X,y):
                augmented = self.augment(image=m, mask=k)
                im.append(augmented['image'])
                mask.append(augmented['mask'])
            return np.array(im) / 255.0, np.array(mask) / 255
			
			
def label_generator(file_path_list_temp, labels, size, img_size, n_channels):
        'Generates data containing batch_size samples'
        # Initialization self.batch_size
        y = np.empty((size, img_size, img_size, 1))
        y_class = []
        for idx, file_path in enumerate(file_path_list_temp[:size]):
            
            id = file_path.split('/')[-1][:-4]
            rle = labels.get(id)
            
            if rle is None:
                mask = np.zeros((1024, 1024))
            else:
                if len(rle) == 1:
                    mask = rle2mask(rle[0], 1024, 1024).T
                else: 
                    mask = np.zeros((1024, 1024))
                    for r in rle:
                        mask =  mask + rle2mask(r, 1024, 1024).T
            
            # Store class
            y[idx,] = cv2.resize(mask,(img_size,img_size))[..., np.newaxis]
            y[y>0] = 255
    
        return  np.array(y) / 255  
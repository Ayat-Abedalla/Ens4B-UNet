# -*- coding: utf-8 -*-

import cv2
import pydicom
import numpy as np
import pandas as pd
from glob import glob
from dataset import parse_dicom

from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm

import keras
import segmentation_models as sm
keras.backend.set_image_data_format('channels_last')

from mask_functions import mask2rle

test_file = 'stage2_siim_data/stage_2_images/*.dcm'
rle_file = 'stage2_siim_data/stage_2_train.csv'

def get_test(test_size, test_metadata_df, img_size ,channels):   
  X = np.empty((test_size, img_size, img_size, channels))
  id_test = []
  #temp = test_metadata_df[2136:3205]  #0, 1068, 2136, 3205
  #temp = temp.reset_index(drop=True)
  for idx, row in test_metadata_df.iterrows(): #temp.iterrows()
    # Store sample
    pixel_array = pydicom.read_file(row['file_path']).pixel_array
    
    image_resized = cv2.resize(pixel_array, (img_size, img_size))

    image_resized = np.repeat(image_resized[...,None],3,2)
    
    id_test.append(str(row['id']))
    
    X[idx,] = image_resized

  return np.array(X) / 255.0
  
def get_model():
    BACKBONE = 'resnet34'
    model = sm.Unet(backbone_name=BACKBONE, encoder_weights='imagenet')
    
    print('Using swa weight model')
    model.load_weights('stage2_model_output/256_resnet34.model')
    
    return model

def prepare_test(test_file, rle_file):
    
    test_dcm  = sorted(glob(test_file))
    print("Number of testing files", len(test_dcm))
    
    # Load train rles
    train_rle = pd.read_csv(rle_file)
    
    test_metadata_df  = parse_dicom(test_dcm, train_rle, encoded_pixels=False)
    
    return test_metadata_df
    
def get_prediction(model, test_data, batch_size):
    preds_test_orig = model.predict(test_data, batch_size=batch_size)
    x_test = np.array([np.fliplr(x) for x in test_data])
    preds_test_flipped = model.predict(x_test, batch_size=batch_size)
    preds_test_flipped = np.array([np.fliplr(x) for x in preds_test_flipped])
    preds_test = 0.5*preds_test_orig + 0.5*preds_test_flipped
    
    return preds_test

def get_rles(preds_test, b_th, r_th):
    rles = []
    i,max_img = 1,10
    plt.figure(figsize=(16,4))

    for p in tqdm(preds_test):
        p = p.squeeze()
        im = cv2.resize(p,(1024,1024))
        im = (im > b_th) 
    
        #zero out the smaller regions.
        if im.sum()< r_th:
            im[:] = 0
    
        im = (im.T*255).astype(np.uint8) 

        rles.append(mask2rle(im, 1024, 1024))
    
        i += 1
        if i<max_img:
            plt.subplot(1,max_img,i)
            plt.imshow(im)
            plt.axis('off')
            
    return rles

def predict_result_val(model,validation_generator,img_size): 
    preds_test = model.predict_generator(validation_generator).reshape(-1, img_size, img_size)
    return preds_test
    
    
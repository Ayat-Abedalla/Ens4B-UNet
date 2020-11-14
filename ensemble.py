# -*- coding: utf-8 -*-

import numpy as np
from glob import glob
import pandas as pd 

from predict import get_rles

def get_pred_file(file_path):
    
    decompressed_array= np.load(file_path)  
    pred = decompressed_array['array1']   
    
    return pred
    
if __name__ == '__main__':
    
    resnet50_512_pred_test = get_pred_file("process_data/val_pre/resnet50_512_pred_test.npz") 
    
    densenet169_preds_test = get_pred_file("process_data/val_pre/dense169_preds_valid.npz") 

    seresnext50_512_pred_test = get_pred_file("process_data/val_pre/seresnext50_512_pred_test.npz") 

    efficientb4_512_preds_test = get_pred_file("process_data/val_pre/efficientb4_512_preds_test.npz") 
    
    ensemble_pred_test =  0.1*densenet169_preds_test + 0.1*resnet50_512_pred_test + 0.4*efficientb4_512_preds_test + 0.4*seresnext50_512_pred_test
    
    test_fn = sorted(glob('stage2_siim_data/stage_2_images/*.dcm'))
    test_IDs = [o.split('/')[-1][:-4] for o in test_fn]
    
    rles = get_rles(ensemble_pred_test, b_th = 0.5, r_th = 2048)
    
    sub_df = pd.DataFrame({'ImageId': test_IDs, 'EncodedPixels': rles})
    sub_df.loc[sub_df.EncodedPixels=='', 'EncodedPixels'] = '-1'
    sub_df.head()
    
    sub_df.to_csv('model_submission/ensemble_submission.csv', index=False)
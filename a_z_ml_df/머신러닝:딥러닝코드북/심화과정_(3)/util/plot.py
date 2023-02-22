"""
Plot result
"""
import numpy as np
import matplotlib.pyplot as plt

def show_mask(imgs_valid,imgs_mask_valid,imgs_mask_predict,sample_number):
    for i in range(0,sample_number):
        plt.figure(figsize=(12, 3))
        plt.subplot(131).imshow(imgs_valid[i,:,:,0],cmap='gray')
        plt.subplot(131).set_title("Validation Image")
        plt.axis('off')
        plt.subplot(132).imshow(imgs_mask_valid[i,:,:,0],cmap='gray')
        plt.subplot(132).set_title("Validation Mask")
        plt.axis('off')
        plt.subplot(133).imshow(imgs_mask_predict[i,:,:,0],cmap='gray')
        plt.subplot(133).set_title("Prediction")
        plt.axis('off')
        plt.show()
        
def show_segmentation(imgs_valid,imgs_mask_valid,imgs_mask_predict,sample_number):
    for i in range(0,sample_number):
        plt.figure(figsize=(12, 3))
        plt.subplot(131).imshow(imgs_valid[i,:,:,0],cmap='gray')
        plt.subplot(131).set_title("Validation Image")
        plt.axis('off')
        plt.subplot(132).imshow(imgs_valid[i,:,:,0]*imgs_mask_valid[i,:,:,0],cmap='gray')
        plt.subplot(132).set_title("Segmentation(Ground Truth)")
        plt.axis('off')
        plt.subplot(133).imshow(imgs_valid[i,:,:,0]*imgs_mask_predict[i,:,:,0],cmap='gray')
        plt.subplot(133).set_title("Segmentation(Prediction)")
        plt.axis('off')
        plt.show()
import os
import numpy as np
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from glob import glob
from util.function import *
def save_dataset(preprocessed_data_path,dataset_path):
    makedir(dataset_path)
    file_list=glob(preprocessed_data_path+"/lungmask_*.npy")
    out_images = []      #final set of images
    out_nodemasks = []   #final set of nodemasks
    for fname in file_list:
        imgs_to_process = np.load(fname.replace("lungmask","image"))
        masks = np.load(fname)
        node_masks = np.load(fname.replace("lungmask","mask"))
        for i in range(len(imgs_to_process)):
            mask = masks[i]
            node_mask = node_masks[i]
            img = imgs_to_process[i]
            new_size = [512,512]   #scaling back up to the original size of the image
            ##apply lung mask##
            img= mask*img          
            # renormalizing the masked image (in the mask region)
            new_mean = np.mean(img[mask>0])  
            new_std = np.std(img[mask>0])
            #  Pulling the background color up to the lower end of the pixel range for the lungs
            old_min = np.min(img)
            img[img==old_min] = new_mean-1.2*new_std 
            img = img-new_mean
            img = img/new_std
            #make image bounding box  (min row, min col, max row, max col)
            labels = measure.label(mask)
            regions = measure.regionprops(labels)
            # Finding the global min and max row over all regions
            min_row = 512
            max_row = 0
            min_col = 512
            max_col = 0
            for prop in regions:
                B = prop.bbox
                if min_row > B[0]:
                    min_row = B[0]
                if min_col > B[1]:
                    min_col = B[1]
                if max_row < B[2]:
                    max_row = B[2]
                if max_col < B[3]:
                    max_col = B[3]
            width = max_col-min_col
            height = max_row - min_row
            if width > height:
                max_row=min_row+width
            else:
                max_col = min_col+height
            img = img[min_row:max_row,min_col:max_col]
            mask =  mask[min_row:max_row,min_col:max_col]
            if max_row-min_row <5 or max_col-min_col<5:
                pass
            else:
                # moving range to -1 to 1 to accomodate the resize function
                mean = np.mean(img)
                img = img - mean
                min = np.min(img)
                max = np.max(img)
                img = img/(max-min)
                new_img = resize(img,[512,512])
                new_node_mask = resize(node_mask[min_row:max_row,min_col:max_col],[512,512])
                idx=np.where(new_node_mask>0)
                new_node_mask[idx]=1
                out_images.append(new_img)
                out_nodemasks.append(new_node_mask)
    num_images = len(out_images)

    #  Writing out images and masks as 1 channel arrays for input into network
    final_images = np.ndarray([num_images,512,512,1],dtype=np.float32)
    final_masks = np.ndarray([num_images,512,512,1],dtype=np.float32)
    for i in range(num_images):
        final_images[i,:,:,0] = out_images[i]
        final_masks[i,:,:,0] = out_nodemasks[i]

    rand_i = np.random.choice(range(num_images),size=num_images,replace=False)
    valid_i = int(0.2*num_images)
    #save train and validation data
    np.save(dataset_path+"train_image.npy",final_images[rand_i[valid_i:]])
    np.save(dataset_path+"train_mask.npy",final_masks[rand_i[valid_i:]])
    np.save(dataset_path+"valid_image.npy",final_images[rand_i[:valid_i]])
    np.save(dataset_path+"valid_mask.npy",final_masks[rand_i[:valid_i]])

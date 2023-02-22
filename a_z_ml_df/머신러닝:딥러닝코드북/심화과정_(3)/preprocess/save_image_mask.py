import os
import SimpleITK as sitk

import numpy as np
import csv
from glob import glob
import pandas as pd
from util.function import *
from preprocess.preprocess_csv import createmask
try:

    from tqdm import tqdm

except:

    print('TQDM does make much nicer wait bars...')

    tqdm = lambda x: x

def save_npy(annotation_path,luna_subset_path,save_path):
    makedir(save_path)
    file_list=glob(luna_subset_path+"*.mhd")
    df_node = pd.read_csv(annotation_path)

    df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))

    df_node = df_node.dropna()
    
    for fcount, img_file in enumerate(tqdm(file_list)):

        mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file

        if mini_df.shape[0]>0:

            # load the data once

            itk_img = sitk.ReadImage(img_file) 

            img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)

            num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
            origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
            spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
            for node_idx, cur_row in mini_df.iterrows():       

                node_x = cur_row["coordX"]

                node_y = cur_row["coordY"]

                node_z = cur_row["coordZ"]
                diam = cur_row["diameter_mm"]

                # keep 3 slices

                imgs = np.ndarray([3,height,width],dtype=np.float32)

                masks = np.ndarray([3,height,width],dtype=np.uint8)

                center = np.array([node_x, node_y, node_z])   # nodule center

                v_center = np.rint((center-origin)/spacing)  # nodule center in voxel space (still x,y,z ordering)
                for i, i_z in enumerate(np.arange(int(v_center[2])-1,

                                 int(v_center[2])+2).clip(0, num_z-1)): # clip prevents going out of bounds in Z

                    mask = createmask(center, diam, i_z*spacing[2]+origin[2],

                                     width, height, spacing, origin)

                    masks[i] = mask

                    imgs[i] = img_array[i_z]
                #save preprocessed image and mask
                np.save(os.path.join(save_path,"image_%04d_%04d.npy" % (fcount, node_idx)),imgs)
                np.save(os.path.join(save_path,"mask_%04d_%04d.npy" % (fcount, node_idx)),masks)

            

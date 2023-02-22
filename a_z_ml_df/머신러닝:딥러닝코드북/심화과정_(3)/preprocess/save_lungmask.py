import numpy as np
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from glob import glob
    
    
def save_lungnpy(preprocessed_data_path):
    file_list=glob(preprocessed_data_path+"image_*.npy")
    for img_file in file_list:
        imgs_to_process = np.load(img_file).astype(np.float64) 
        for i in range(len(imgs_to_process)):
            img = imgs_to_process[i]
            #Standardize the pixel values
            mean = np.mean(img)
            std = np.std(img)
            img = img-mean
            img = img/std
            # Find the average pixel value near the lungs
            middle = img[100:400,100:400] 
            mean = np.mean(middle)  
            max = np.max(img)
            min = np.min(img)
            img[img==max]=mean
            img[img==min]=mean
            
            # Using Kmeans to separate foreground and background
            kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
            centers = sorted(kmeans.cluster_centers_.flatten())
            threshold = np.mean(centers)
            thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
            eroded = morphology.erosion(thresh_img,np.ones([4,4]))
            dilation = morphology.dilation(eroded,np.ones([10,10]))
            labels = measure.label(dilation)
            label_vals = np.unique(labels)
            regions = measure.regionprops(labels)
            good_labels = []
            for prop in regions:
                B = prop.bbox
                if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
                    good_labels.append(prop.label)
            mask = np.ndarray([512,512],dtype=np.int8)
            mask[:] = 0
            for N in good_labels:
                mask = mask + np.where(labels==N,1,0)
            mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
            imgs_to_process[i] = mask
        #save lung mask
        np.save(img_file.replace("image","lungmask"),imgs_to_process)

            

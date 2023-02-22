import os
import SimpleITK as sitk

import numpy as np

import csv

from glob import glob

import pandas as pd

def matrix2int16(matrix):

    ''' 

Matrix must be a numpy array NXN

Returns uint16 version

    '''

    m_min= np.min(matrix)

    m_max= np.max(matrix)

    matrix = matrix-m_min

    return(np.array(np.rint( (matrix-m_min)/float(m_max-m_min) * 65535.0),dtype=np.uint16))
        
def makedir(path):
    """
    Make directory
    """
    os.makedirs(path, exist_ok=True)
    
def get_filename(file_list, case):
    for f in file_list:
        
        if case in f:

            return(f)
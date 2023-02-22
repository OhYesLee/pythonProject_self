import imghdr
import math
import os
import re
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, UpSampling2D, Activation,concatenate, BatchNormalization
from tensorflow.keras.optimizers import SGD

def UNet():
    row_size = 512
    col_size = 512
    inputs = tf.keras.Input(shape=(row_size, col_size,1))
    conv1 = Conv2D(kernel_size=(3,3), filters=32, padding='same')(inputs)
    conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(kernel_size=(3,3), filters=32, padding='same')(conv1)
    conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(kernel_size=(3,3), filters=32, padding='same')(conv1)
    conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(kernel_size=(3,3), filters=64,padding='same')(pool1)
    conv2 = BatchNormalization(axis=3)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(kernel_size=(3,3), filters=64,padding='same')(conv2)
    conv2 = BatchNormalization(axis=3)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(kernel_size=(3,3), filters=64,padding='same')(conv2)
    conv2 = BatchNormalization(axis=3)(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(kernel_size=(3,3), filters=128,padding='same')(pool2)
    conv3 = BatchNormalization(axis=3)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(kernel_size=(3,3), filters=128,padding='same')(conv3)
    conv3 = BatchNormalization(axis=3)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(kernel_size=(3,3), filters=128,padding='same')(conv3)
    conv3 = BatchNormalization(axis=3)(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(kernel_size=(3,3), filters=256,padding='same')(pool3)
    conv4 = BatchNormalization(axis=3)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(kernel_size=(3,3), filters=256,padding='same')(conv4)
    conv4 = BatchNormalization(axis=3)(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(kernel_size=(3,3), filters=512,padding='same')(pool4)
    conv5 = BatchNormalization(axis=3)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(kernel_size=(3,3), filters=512,padding='same')(conv5)
    conv5 = BatchNormalization(axis=3)(conv5)
    conv5 = Activation('relu')(conv5)
    
    up6=concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(kernel_size=(3,3), filters=256, padding='same')(up6)
    conv6 = BatchNormalization(axis=3)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(kernel_size=(3,3), filters=256,padding='same')(conv6)
    conv6 = BatchNormalization(axis=3)(conv6)
    conv6 = Activation('relu')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(kernel_size=(3,3), filters=128,padding='same')(up7)
    conv7 = BatchNormalization(axis=3)(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(kernel_size=(3,3), filters=128,padding='same')(conv7)
    conv7 = BatchNormalization(axis=3)(conv7)
    conv7 = Activation('relu')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(kernel_size=(3,3), filters=64,padding='same')(up8)
    conv8 = BatchNormalization(axis=3)(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(kernel_size=(3,3), filters=64,padding='same')(conv8)
    conv8 = BatchNormalization(axis=3)(conv8)
    conv8 = Activation('relu')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(kernel_size=(3,3), filters=32,padding='same')(up9)
    conv9 = BatchNormalization(axis=3)(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(kernel_size=(3,3), filters=32,padding='same')(conv9)
    conv9 = BatchNormalization(axis=3)(conv9)
    conv9 = Activation('relu')(conv9)

    conv10 = Conv2D(kernel_size=(1,1), filters=1,activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    return model
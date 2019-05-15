from __future__ import print_function
import numpy as np
import keras
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras.preprocessing.image import *
from keras.regularizers import *
from keras import backend as K
from keras.models import Model
from scipy.io import loadmat
import os
from datetime import datetime
from tqdm import tqdm

def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1

def mk_dir(dir):
    try:
        os.mkdir( dir )
    except OSError:
        pass

def rgb2gray(rgb):
    return np.expand_dims(np.dot(rgb[...,:3], [0.299, 0.587, 0.114]),-1)

def loadWiki():
    mat = loadmat("./dataset/wiki.mat")
    lista = mat['wiki'][0][0]
    full_path = lista[2][0]
    age = np.array([calc_age(x,y) for x,y in zip(lista[1][0],lista[0][0]) ])
    gender = lista[3][0] #0 for female and 1 for male, NaN if unknown
    del lista 
    paths = []
    label   = [] 
    print("Loading dataset...")
    for i in tqdm(range(len(gender))):
        if str(gender[i]) != 'nan':
            path = './dataset/'+str(full_path[i][0])
            path = path.replace(':','_')
            img = cv2.imread(path)
            if img is not None and img.shape != (1,1) :
                label.append((gender[i],age[i]))
                paths.append(path)
            del img
    np.save("data_paths_v2.npy",np.array(paths))
    np.save("data_label_v2.npy",np.array(label))


def cnn(input_shape,num_gender=2,num_ages=100,gn = 0.2,kernel = (3,3)):
    # Design model
    inputs = Input(shape=input_shape)
    # Block 
    x = BatchNormalization()(inputs)
    x = GaussianNoise(gn)(x)
    x = Conv2D(16, kernel, padding='same') (x)
    x = Conv2D(16, kernel, padding='same') (x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Block 
    x = BatchNormalization()(x)
    x = GaussianNoise(gn)(x)
    x = Conv2D(32, kernel, padding='same') (x)
    x = Conv2D(32, kernel, padding='same') (x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Block 
    x = BatchNormalization()(x)
    x = GaussianNoise(gn)(x)
    x = Conv2D(64, kernel, padding='same') (x)
    x = Conv2D(64, kernel, padding='same') (x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Block 
    x = BatchNormalization()(x)
    x = GaussianNoise(gn)(x)
    x = Conv2D(128, kernel, padding='same') (x)
    x = Conv2D(128, kernel, padding='same') (x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Block 
    x = BatchNormalization()(x)
    x = GaussianNoise(gn)(x)
    x = Conv2D(256, kernel, padding='same') (x)
    x = Conv2D(256, kernel, padding='same') (x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)    

    # Classification block
    y = Flatten()(x)

    pred_g_softmax = Dense(num_gender, activation='softmax', kernel_initializer='he_normal', name='gender')(y)
    pred_a_softmax = Dense(num_ages, activation='softmax', kernel_initializer='he_normal', name='age')(y)
    # Instantiate model.
    model = Model(inputs=inputs, outputs=[pred_g_softmax, pred_a_softmax])
    return model

    return model
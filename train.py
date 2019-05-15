from keras.preprocessing.image import img_to_array
from keras.models import load_model
from scipy.io import loadmat
import numpy as np
import argparse
import cv2
import os
import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda,Cropping2D,Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization as BN
from keras.layers import GaussianNoise as GN
from keras.optimizers import *
from keras.callbacks import TensorBoard,LearningRateScheduler,EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from keras.initializers import glorot_uniform
from utils import *
from random import randint
from tqdm import tqdm
import string
from DataGenerator import *
import random


# Parameters
params = {'dim': (32,32),
          'n_channels': 1,
          'batch_size': 2048,
          'n_gender': 2,
          'n_ages': 100,
          'shuffle': True,
          }

EPOCH = 50
FINAL_WEIGHTS_PATH='final_weights.hdf5'
# Datasets
data_path   = np.load("data_paths_v2.npy")
data_labels = np.load("data_labels_v2.npy")
lman   = [x for x,y in zip(data_path,data_labels) if int(y[0]) == 1]
lwoman = [x for x,y in zip(data_path,data_labels) if int(y[0]) == 0]
eman   = [x for x in data_labels if int(x[0]) == 1]
ewoman = [x for x in data_labels if int(x[0]) == 0]


partition = []
labels    = []
##print('Loading dataset...')
for i in tqdm(range(len(data_path))):
    man = randint(0,len(lman)-1)
    woman = randint(0,len(lwoman)-1)
    manImg = cv2.resize(cv2.cvtColor(cv2.imread(lman[man]), cv2.COLOR_BGR2GRAY),params['dim'])
    womanImg = cv2.resize(cv2.cvtColor(cv2.imread(lwoman[woman]), cv2.COLOR_BGR2GRAY),params['dim'])
    partition.append(np.expand_dims(manImg,-1))
    partition.append(np.expand_dims(womanImg,-1))
    labels.append(eman[man])
    labels.append(ewoman[woman])

del lman
del lwoman
del eman
del ewoman
del data_labels
del data_path


partition = np.asarray(partition)/255
gender = np.asarray([y[0] for y in labels])
ages = np.asarray([y[1] for y in labels])
valid_age_range = np.isin(ages, [x for x in range(100)])
gender = gender[valid_age_range]
ages = ages[valid_age_range].astype(int)
partition = partition[valid_age_range]


gender = keras.utils.to_categorical(gender, num_classes=params['n_gender']).astype(int)
age_bins = np.linspace(0, 100, params['n_ages'])
age_step = np.digitize(ages, age_bins)
ages = keras.utils.to_categorical(age_step, params['n_ages'])

x_train,x_test,y_train_gender,y_test_gender,y_train_age,y_test_age = train_test_split(partition,gender,ages, test_size=0.2)
x_train,x_val,y_train_gender,y_val_gender,y_train_age,y_val_age = train_test_split(x_train,y_train_gender,y_train_age, test_size=0.2)

xx=keras.utils.to_categorical(0, num_classes=params['n_gender'])
xy=keras.utils.to_categorical(1, num_classes=params['n_gender'])
nfemale = len([i for i in y_train_gender if all(i == xx)]) 
nmen    = len([i for i in y_train_gender if all(i == xy)])
print("Number of man: {} , woman {}".format(nmen,nfemale))

# class_weight = {0 : nfemale/len(y_train_gender),
#                 1 : nmen/len(y_train_gender)}
# print(class_weight)
print("x_train:{}, y_train_gender:{}, y_train_age:{}".format(x_train.shape,y_train_gender.shape,y_train_age.shape))
print("x_test:{},y_test_gender:{}, y_test_age:{}".format(x_test.shape,y_test_gender.shape,y_test_age.shape))
print("x_val:{}, y_val_gender:{}, y_val_age:{}".format(x_val.shape,y_val_gender.shape,y_val_age.shape))


training_generator   = DataGenerator(x_train, y_train_gender, y_train_age, **params)
test_generator       = DataGenerator(x_test,  y_test_gender,  y_test_age,  **params)
validation_generator = DataGenerator(x_val,   y_val_gender,   y_val_age,   **params)


model = cnn((*params['dim'],1),num_ages=params['n_ages'],num_gender=params['n_gender'])
model.summary()

## OPTIM AND COMPILE
#optimizer
optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-06, decay=0.0)

# load weights
print("Preload weights...")
## load weights
filepath="./"+FINAL_WEIGHTS_PATH
exists = os.path.isfile(filepath)
if exists:
    model.load_weights(filepath)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics={'gender': 'accuracy',
             'age': 'accuracy'},
)


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > EPOCH*0.8:
        lr *= 0.5e-3
    elif epoch > EPOCH*0.6:
        lr *= 1e-3
    elif epoch > EPOCH*0.4:
        lr *= 1e-2
    elif epoch > EPOCH*0.2:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

callbacks = [
    LearningRateScheduler(lr_schedule),
    ReduceLROnPlateau(verbose=1, epsilon=0.001, patience=4),
    ModelCheckpoint(
        os.path.join('checkpoints', 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min"
    ),
    
]

## TRAINING
history=model.fit_generator(generator = training_generator,
                            steps_per_epoch=len(x_train)//params['batch_size'],
                            validation_data=validation_generator,
                            validation_steps=len(x_val)//params['batch_size'],
                            epochs=EPOCH,
                            callbacks=callbacks,
                            max_queue_size=3,
                            verbose=1)

print("Saving weights...")
model.save(os.path.join(".", "Age-Gender_model.h5"))
model.save_weights(os.path.join(".", FINAL_WEIGHTS_PATH), overwrite=True)

score = model.evaluate_generator(test_generator,
                                steps=len(x_test)//params['batch_size'],
                                verbose=1)

print('Test score:', score)

#OUTPUT

plt.plot(history.history['age_acc'])
plt.plot(history.history['gender_acc'])
plt.plot(history.history['val_gender_acc'])
plt.plot(history.history['val_age_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['age_acc', 'gender_acc','val_gender_acc','val_age_acc'], loc='upper left')
plt.show()


plt.plot(history.history['age_loss'])
plt.plot(history.history['gender_loss'])
plt.plot(history.history['val_gender_loss'])
plt.plot(history.history['val_age_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['age_loss', 'gender_loss','val_gender_loss','val_age_loss'], loc='upper left')
plt.show()

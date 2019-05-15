import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, list_IDs_gender,list_IDs_age, batch_size=32, dim=(32,32), n_channels=1,
                 n_ages=100,n_gender=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.data = data
        self.list_IDs_gender = list_IDs_gender
        self.list_IDs_age = list_IDs_age
        self.n_channels = n_channels
        self.n_ages = n_ages
        self.n_gender = n_gender
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        data   = np.asarray([x for x in self.data[indexes]])
        gender = np.asarray([x for x in self.list_IDs_gender[indexes]])
        age    = np.asarray([x for x in self.list_IDs_age[indexes]])

        datagen = ImageDataGenerator(
            width_shift_range=0.2,
            height_shift_range=0.2,
            rotation_range = 8,
            vertical_flip=True)
        datagen.fit(data)
        return data, {'gender': gender,'age': age}

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs_gender))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset=True)
from keras.utils.vis_utils import plot_model
from nnv import NNV

###https://keras.io/examples/

class CNN:
    def __init__(self):
        pass

    def split_and_normalize_data(self, data, DataType='Mnist'):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = data
        self.X_train, self.X_test = self.X_train / 255.0, self.X_test/ 255.0 #normalize_data
        print('X_train.shape:\n', self.X_train.shape, '\ny_train.shape:\n', self.y_train.shape, '\nX_test.shape:\n', self.X_test.shape, '\ny_test.shape:\n', self.y_test.shape)
        if DataType=='Mnist':
            # self.X_train = tf.reshape(self.X_train, [self.X_train.shape[0], 28, 28, 1]) #from N * 28 * 28 to N * 28 * 28 * 1
            self.X_train = np.expand_dims(self.X_train, -1)
            # self.X_test = tf.reshape(self.X_test, [self.X_test.shape[0], 28, 28, 1]) #from N * 28 * 28 to N * 28 * 28 * 1
            self.X_test = np.expand_dims(self.X_test, -1)
            # self.y_train = tf.keras.layers.Flatten()(self.y_train)
            self.y_train = self.y_train.flatten()
            #self.y_test = tf.keras.layers.Flatten()(self.y_test)
            self.y_test = self.y_test.flatten()
            print('max(self.y_train)', max(self.y_train))
            print('max(self.y_test)', max(self.y_test))
        elif DataType == 'CIFAR':
            pass
        else:
            print('Error with Data type')
        print('X_train.shape:\n', self.X_train.shape, '\ny_train.shape:\n', self.y_train.shape, '\nX_test.shape:\n', self.X_test.shape, '\ny_test.shape:\n', self.y_test.shape)


    def create_neural_network(self):
        print('self.X_train[0].shape:', self.X_train[0].shape)
        i = tf.keras.Input(shape=self.X_train[0].shape)#  self.X_train[0].shape = (28, 28, 1)
        x = tf.keras.layers.Conv2D(32, (3, 3), strides=2, activation='relu')(i) #padding=same/valid/full
        x = tf.keras.layers.Conv2D(64, (3, 3), strides=2, activation='relu')(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), strides=2, activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        ######--------------------here convolotion stop and start NN---------------------------------------------------
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        print('len(np.unique(self.y_train)):', len(np.unique(self.y_train)))
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(len(np.unique(cnn.y_train)), activation='softmax')(x)
        self.model = tf.keras.models.Model(i, x)

    def run_model(self, optimizer='adam', loss='sparse_categorical_crossentropy', batch_size=32, n_epochs=10):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        self.model.fit(x=self.X_train, y=self.y_train, validation_data=(self.X_test, self.y_test), epochs=n_epochs, batch_size=batch_size)

        pass

    def epochs_graph(self):
       pass

    def predict(self):
        pass

    def accuracy(self):
        pass

    def save_and_load_model(self):
        pass


if __name__ == '__main__':
    MNIST_data = tf.keras.datasets.fashion_mnist.load_data()  # N*28*28 (grayscale)
    CIFAR10_data = tf.keras.datasets.cifar10.load_data()  # N*32*32*3 (color image)


    data = MNIST_data
    cnn = CNN()
    cnn.split_and_normalize_data(data)
    cnn.create_neural_network()
    cnn.run_model()
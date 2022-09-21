import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, add, Input, Dense, Flatten
from tensorflow.keras.models import Model
from nnv import NNV
from PIL import Image
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.vis_utils import plot_model
from scipy import ndimage, misc
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset=True)

###https://keras.io/examples/
####https://www.kaggle.com/getting-started/253300
###https://transcranial.github.io/keras-js/#/
### dropout explain: https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-dropout-with-keras.md

class CNN:
    def __init__(self):
        pass

    def split_and_normalize_data(self, data):
        self.data_size = data[0][0].shape[1:]
        (self.X_train, self.y_train), (self.X_test, self.y_test) = data
        self.X_train, self.X_test = self.X_train / 255.0, self.X_test / 255.0 #normalize_data
        # self.y_train = tf.keras.layers.Flatten()(self.y_train)
        self.y_train = self.y_train.flatten()
        # self.y_test = tf.keras.layers.Flatten()(self.y_test)
        self.y_test = self.y_test.flatten()

        if self.data_size == (28, 28):
            self.data_type = 'Mnist'
            # self.X_train = tf.reshape(self.X_train, [self.X_train.shape[0], 28, 28, 1]) #from N * 28 * 28 to N * 28 * 28 * 1
            self.X_train = np.expand_dims(self.X_train, -1)
            # self.X_test = tf.reshape(self.X_test, [self.X_test.shape[0], 28, 28, 1]) #from N * 28 * 28 to N * 28 * 28 * 1
            self.X_test = np.expand_dims(self.X_test, -1)
        elif self.data_size == (32, 32, 3):
            self.data_type = 'CIFAR10'
        else:
            print('data are not fit to Mnist(28, 28) or CIFAR(32, 32, 3)')
        #print('X_train.shape:\n', self.X_train.shape, '\ny_train.shape:\n', self.y_train.shape, '\nX_test.shape:\n', self.X_test.shape, '\ny_test.shape:\n', self.y_test.shape)

    def create_neural_network(self):
        print('self.X_train[0].shape:', self.X_train[0].shape)
        i = Input(shape=self.X_train[0].shape) #self.X_train[0].shape = (28, 28, 1) or (32, 32, 3)
        x = Conv2D(32, (3, 3), strides=2, padding='same', activation='relu')(i) #padding=same/valid/full
        x = Conv2D(64, (3, 3), strides=2, padding='same', activation='relu')(x) # 64 No of filters
        x = Conv2D(128, (3, 3), strides=2, padding='same', activation='relu')(x)
        x = Flatten()(x)
        ######--------------------here convolotion stop and start NN---------------------------------------------------
        #x = tf.keras.layers.Dropout(0.2)(x)
        x = Dense(1024, activation='relu')(x) #Mnist 512
        print('len(np.unique(self.y_train)):', len(np.unique(self.y_train)))
        #x = tf.keras.layers.Dropout(0.2)(x)
        x = Dense(len(np.unique(self.y_train)), activation='softmax')(x)
        self.model = Model(i, x)

    def run_model(self, optimizer='adam', loss='sparse_categorical_crossentropy', batch_size=32, n_epochs=1):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        self.modelFit = self.model.fit(x=self.X_train, y=self.y_train, validation_data=(self.X_test, self.y_test), epochs=n_epochs, batch_size=batch_size)

    def loss_function(self):
        plt.figure()
        plt.title("Lose Function Per Epoch")
        plt.plot(self.modelFit.history['loss'], label='loss')
        plt.plot(self.modelFit.history['val_loss'], label='val_loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss Function")
        plt.legend()
        plt.savefig('Loss_Function_CNN.png')
        im = Image.open('Loss_Function_CNN.png')
        im.show()

    def accuracy(self):
        plt.figure()
        plt.title("accuracy Per Epoch")
        plt.plot(self.modelFit.history['accuracy'], label='acc')
        plt.plot(self.modelFit.history['val_accuracy'], label='val_acc')
        plt.xlabel("Epochs")
        plt.ylabel("Loss Function")
        plt.legend()
        plt.savefig('accuracy_CNN.png')
        im = Image.open('accuracy_CNN.png')
        im.show()

    def predict_test_data(self):
        self.p_test = self.model.predict(self.X_test).argmax(axis=1)

    def confusion_matrix1(self, normalize=False):
        if self.data_type == 'Mnist':
            self.labels = '''T-shirt/top
            Trouser
            Pullover 
            Dress 
            Coat
            Sandal 
            Shirt 
            Sneaker 
            Bag 
            Ankle-boot'''.split()

        elif self.data_type == 'CIFAR10':
            self.labels = '''airplane
            automobile
            bird
            cat
            deer
            dog
            frog
            horse
            ship
            truck'''.split()
        else:
            print('data are not fit to Mnist(28, 28) or CIFAR(32, 32, 3)')

        tick_marks = np.arange(len(self.labels))
        cm = confusion_matrix(self.y_test, self.p_test)

        cmap = plt.get_cmap('Oranges')
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title('Confusion matrix')
        plt.xticks(tick_marks, self.labels, rotation=45)
        plt.yticks(tick_marks, self.labels)
        plt.colorbar()

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylim(len(self.labels) - 0.5, -0.5)
        plt.ylabel('True self.labels')
        plt.xlabel('Predicted self.labels')
        plt.savefig('Confusion_matrix.png', dpi=500, bbox_inches='tight')
        im = Image.open('Confusion_matrix.png')
        im.show()

    def misclassified(self):
        misclassified_index = np.where(self.p_test != self.y_test)[0]
        i = np.random.choice(misclassified_index)
        title = 'True label:' + str(self.labels[self.y_test[i]]) + 'Predicted:' +  str(self.labels[self.y_test[i]])
        plt.figure()
        plt.title(title)
        plt.imshow(self.X_test[i].reshape(self.data_size))
        plt.savefig('misclassified_index.png')
        im = Image.open('misclassified_index.png')
        im.show()

    def visualize_training_data(self):
        fig = plt.figure()
        for i in range(9):
            ran_idx = random.randint(0, len(self.X_train))
            plt.subplot(3, 3, i + 1)
            plt.tight_layout()
            plt.imshow(self.X_train[ran_idx], cmap='gray', interpolation='none')
            plt.title("label: {}".format(self.labels[self.y_train[ran_idx]]))
            plt.xticks([])
            plt.yticks([])
        plt.savefig('visualize_data_training.png', dpi=500, bbox_inches='tight')
        im = Image.open('visualize_data_training.png')
        im.show()

    def import_new_data(self, new_data_path):
        pass

    def predict_new_data(self):
        pass
        #print('self.model.predict(self.import_new_data())', self.model.predict())

    def save_and_load_model(self):
        self.model.save('C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\save_CNN.h5')
        #loaded_model = tf.keras.models.load_model('C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\')

    def visualize_model(self):
        # import netron as nt
        # from ann_visualizer.visualize import ann_viz
        # plt.figure()
        # ann_viz(self.model, view=True, title='CNN_model_ visualize')
        # plt.savefig('CNN_model_ visualize', dpi=500, bbox_inches='tight')
        # im = Image.open('CNN_model_ visualize')
        # im.show()
        #-----------------------------------------#---------------------------------------
        # nt.start('C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\save_CNN.h5', 8081)
        # nt.stop()
        pass


if __name__ == '__main__':
    MNIST_data = tf.keras.datasets.fashion_mnist.load_data()  # N*28*28 (grayscale)
    CIFAR10_data = tf.keras.datasets.cifar10.load_data()  # N*32*32*3 (color image)

    data = CIFAR10_data
    cnn = CNN()
    cnn.split_and_normalize_data(data)
    cnn.create_neural_network()
    cnn.run_model()
    cnn.loss_function()
    cnn.accuracy()
    cnn.predict_test_data()
    cnn.confusion_matrix1()
    cnn.misclassified()
    cnn.import_new_data('dress.JPG')
    cnn.predict_new_data()
    cnn.visualize_training_data()
    cnn.save_and_load_model()
    cnn.visualize_model()

    # model_graph = plot_model(cnn.model, 'my_CNN_graph_model.png', show_shapes=True)
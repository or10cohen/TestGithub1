import pandas as pd
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset = True)


df = pd.read_csv('DATA/fake_reg.csv')
X = df[['feature1', 'feature2']].values
y = df['price'].values

class FirsNeuralNetwork:
    def __init__(self, X, y, n_epochs=250):
        self.X = X
        self.y = y
        self.n_epochs = n_epochs

    def splitAndNormlaizeData(self):
        X_train, X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        #X_train, X_test = MinMaxScaler().fit_transform(X_train), MinMaxScaler().fit_transform(X_test)
        self.scalar = MinMaxScaler()
        self.scalar.fit(X_train)
        self.scalar.fit(X_test)  ##????
        self.X_train, self.X_test = self.scalar.transform(X_train), self.scalar.transform(X_test)

    def creatNeuralNetwork(self):
        ###------------creat NN option1--------------------------
        model1 = tf.keras.Sequential([tf.keras.layers.Dense(4, activation='relu'),
                            tf.keras.layers.Dense(3, activation='relu'),
                            tf.keras.layers.Dense(3, activation='relu'),
                            tf.keras.layers.Dense(1)])                         #output layer
        ###------------creat NN option-----------------------------
        model2 = tf.keras.Sequential()
        model2.add(tf.keras.layers.Dense(4, activation='relu'))
        model2.add(tf.keras.layers.Dense(4, activation='relu'))
        model2.add(tf.keras.layers.Dense(4, activation='relu'))
        model2.add(tf.keras.layers.Dense(1))                                  #output layer
        self.model1, self.model2 = model1, model2

    def runModel(self, optimizer='rmsprop', loss='mse'):
        self.model2.compile(optimizer=optimizer, loss=loss)
        self.model2.fit(x=self.X_train, y=self.y_train, epochs=self.n_epochs)

    def epochsGraph(self):
        ##---------------------graph epochs--------------------------
        axis_x = [i for i in range(self.n_epochs)]
        axis_y = self.model2.history.history['loss']
        plt.plot(axis_x, axis_y)
        plt.savefig('Graph.png')
        im = Image.open('Graph.png')
        # im.show()

    def predict(self):
        ##------------------------predictions------------------------
        test_predict = self.model2.predict(self.X_test)
        test_predict = pd.Series(test_predict.reshape(300,))
        pred_df = pd.DataFrame(self.y_test, columns=['Test True Y'])
        pred_df = pd.concat([pred_df, test_predict], axis=1)
        pred_df.columns = ['Test True Y', 'Model predict']
        print(pred_df)
        ###------------------------predictions on new data without label------------------------
        self.new_gem = [[998,1000]]
        self.new_gem = self.scalar.transform(self.new_gem)                 #scale by the X_train
        print(self.new_gem)
        print(Fore.MAGENTA + 'predictions on new data without label', self.model2.predict(self.new_gem))

    def accuracy(self):
        pass

    def saveAndLoadModel(self):
        ###-----------------------save and load your model----------------------------------------
        self.model2.save('C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\')
        loaded_model = tf.keras.models.load_model('C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\')
        loaded_model.predict(self.new_gem)
        print(Fore.MAGENTA + 'new model predictions on new data without label', loaded_model.predict(self.new_gem))
        if loaded_model.predict(self.new_gem) == self.model2.predict(self.new_gem):
            print(Fore.GREEN + 'the loaded_model = save model. good save and load!')
        else:
            print(Fore.GREEN + 'error load or save model!')



if __name__ == '__main__':
    df = pd.read_csv('DATA/fake_reg.csv')
    X = df[['feature1', 'feature2']].values
    y = df['price'].values
    run = FirsNeuralNetwork(X, y)
    run.splitAndNormlaizeData()
    run.creatNeuralNetwork()
    run.runModel()
    run.epochsGraph()
    run.predict()
    run.saveAndLoadModel()

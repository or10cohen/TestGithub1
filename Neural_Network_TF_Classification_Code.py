import pandas as pd
import numpy as np
from PIL import Image
#import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset=True)


class FirsNeuralNetwork:
    def __init__(self, X, y, n_epochs=250):
        self.X = X
        self.y = y
        self.n_epochs = n_epochs

    def split_and_normalize_data(self):
        X_train, X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        # X_train, X_test = MinMaxScaler().fit_transform(X_train), MinMaxScaler().fit_transform(X_test)
        self.scalar = MinMaxScaler()
        self.scalar.fit(X_train)
        self.scalar.fit(X_test)  ##????
        self.X_train, self.X_test = self.scalar.transform(X_train), self.scalar.transform(X_test)

    def create_neural_network(self):
       pass

    def run_model(self, optimizer='rmsprop', loss='mse'):   #loss classification = softmax(נותן לכל נוירון באווטפוט יחס לתשובה הנכונה, הסכום שווה ל1)
       pass

    def epochs_graph(self):
        pass

    def predict(self):
        pass

    def accuracy(self):
        pass

    def save_and_load_model(self):
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
    run.split_and_normalize_data()
    run.create_neural_network()
    run.run_model()
    run.epochs_graph()
    run.predict()
    run.save_and_load_model()

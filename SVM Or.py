import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset = True)



class svm:
    def __init__(self, chose_dataset, test_size=0.33, random_state=42, kernel='linear', c=1.0):
        self.data_set = chose_dataset
        self.test_size, self.random_state, self.kernel, self.C = test_size, random_state, kernel, c
        self.X, self.y = self.data()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        self.normalize_X_train, self.normalize_X_test = self.normalize_data()
        self.model = self.fit()
        self.confusion_matrixes, self.classification_reports = self.predict()

    def __str__(self):
        print(Fore.RED + '\ndataset.keys()\n', self.data_set.keys())
        print(Fore.RED + '\ndataset.filename\n', self.data_set.filename)
        print(Fore.RED + '\ndataset.data[:5, :5]\n', self.data_set.data[:5, :5])
        print(Fore.RED + '\ndataset.data shape\n', self.data_set.data.shape)
        print(Fore.RED + '\ndataset.target[:20]\n', self.data_set.target[:20])
        print(Fore.RED + '\ndataset.target shape\n', self.data_set.target.shape)

    def data(self):
        X = self.data_set.data
        y = self.data_set.target
        return X, y

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size,
                                                                random_state=self.random_state)
        return X_train, X_test, y_train, y_test

    def normalize_data(self):
        normalize_train_data, normalize_test_data = MinMaxScaler().fit_transform(
            self.X_train), MinMaxScaler().fit_transform(self.X_test)
        return normalize_train_data, normalize_test_data

    def fit(self):
        model = SVC()
        model.fit(self.normalize_X_train, self.y_train)
        return model

    def predict(self):
        predict_on_model = self.model.predict(self.normalize_X_test)
        confusion_matrixes = confusion_matrix(self.y_test, predict_on_model)
        classification_reports = classification_report(self.y_test, predict_on_model)
        return confusion_matrixes, classification_reports


if __name__ == '__main__':
    are_dataset = datasets.load_breast_cancer()
    run_svm = svm(are_dataset)
    # run_svm.__str__()

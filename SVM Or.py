import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset = True)



class svm:
    def __init__(self, chose_dataset, test_size=0.33, random_state=42, kernel='linear', c=1.0):
        self.data_set = chose_dataset
        self.test_size, self.random_state, self.kernel, self.C = test_size, random_state, kernel, c
        self.X, self.y = self.data()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        # self.normalize_train_data, self.normalize_test_data = self.normalize_data()
        # self.y_predict, self.a, self.b, self.r, self.t, self.b_up_margin, self.b_down_margin = self.fit()

    def __str__(self):
        print(Fore.RED + 'cancer.keys()\n', self.data_set.keys())
        print(Fore.RED + '\ncancer.data\n', self.data_set.data[:5, :5])
        print(Fore.RED + 'shape\n', self.data_set.data.shape)
        print(Fore.RED + '\ncancer.target\n', self.data_set.target[:20])
        print(Fore.RED + 'shape\n', self.data_set.target.shape)

    def data(self):
        X = self.data_set.data
        y = self.data_set.target
        return X, y

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size,
                                                                random_state=self.random_state)
        # y_train = np.where(y_train == 2, 1, y_train)
        return X_train, X_test, y_train, y_test

    def fit(self):
        model = SVC()
        model.fit(self.X_train, self.y_train)

are_dataset = datasets.load_breast_cancer()
run_svm = svm(are_dataset)
run_svm.fit()
print(run_svm.fit())
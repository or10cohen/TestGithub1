import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler


class svm:
    def __init__(self, chose_dataset, test_size=0.05, random_state=42, kernel='linear', c=1.0):
        self.data_set = chose_dataset
        self.test_size, self.random_state, self.kernel, self.C = test_size, random_state, kernel, c
        self.X, self.y = self.data()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        self.normalize_train_data, self.normalize_test_data = self.normalize_data()
        self.y_predict, self.a, self.b, self.r, self.t, self.b_up_margin, self.b_down_margin = self.fit()

    def data(self):
        X = self.data_set.data[:, :2]
        y = self.data_set.target
        return X, y

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size,
                                                            random_state=self.random_state)
        y_train = np.where(y_train == 2, 1, y_train)
        return X_train, X_test, y_train, y_test

    def normalize_data(self):
        normalize_train_data, normalize_test_data = MinMaxScaler().fit_transform(
            self.X_train), MinMaxScaler().fit_transform(self.X_test)
        return normalize_train_data, normalize_test_data

    def fit(self):
        clf = SVC(C=self.C, kernel=self.kernel)
        clf.fit(self.normalize_train_data, self.y_train)
        y_predict = clf.predict(self.normalize_test_data[:, :].reshape(-1, 2))
        print()
        w = clf.coef_
        a = -w[0][0] / w[0][1]
        support_vectors = clf.support_vectors_
        b = -clf.intercept_ / w[0][1]
        # print(clf.coef_)
        # print(clf.intercept_)
        r = np.linspace(0, 1, 1000)
        t = a * r + b
        b_up_margin = -(clf.intercept_ + 1) / w[0][1]
        b_down_margin = -(clf.intercept_ - 1) / w[0][1]
        return y_predict, a, b, r, t, b_up_margin, b_down_margin

    def print_data(self):
        up_margin = self.a * self.r + self.b_up_margin
        down_margin = self.a * self.r + self.b_down_margin
        plt.scatter(self.normalize_train_data[:, 0], self.normalize_train_data[:, 1])
        plt.plot(self.r, self.t, color='k')
        plt.plot(self.r, up_margin, color='r')
        plt.plot(self.r, down_margin, color='r')
        plt.savefig('C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\SVM.png')


if __name__ == '__main__':
    chose_dataset = datasets.load_iris()
    SVM = svm(chose_dataset)
    SVM.print_data()
    plt.show()


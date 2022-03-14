import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

# Load iris data and labels
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target



class SVM:

    def __init__(self, X, y, test_size=0.05, random_state=42, kernel='linear', C=1.0):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.kernel = kernel
        self.C = C
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        self.normalize_train_data, self.normalize_test_data = self.normalize_data()
    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)
        y_train = np.where(y_train == 2, 1, y_train)
        return X_train, X_test, y_train, y_test

    def  normalize_data(self):
        normalize_train_data = MinMaxScaler.fit_transform(self.X_train)
        normalize_test_data = MinMaxScaler.fit_transform(self.X_test)
        return normalize_train_data, normalize_test_data

    def print_data(self):
    # print(normalize_data[:10, :10])
    clf = SVC(C=self.C, kernel=self.kernel)
    clf.fit(self.normalize_train_data, self.y_train)
    y_predict = clf.predict(normalize_test_data[:, :].reshape(-1, 2))
    print()



w = clf.coef_
# b = clf.intercept_
support_vectors = clf.support_vectors_
b = -clf.intercept_/w[0][1]
# print(clf.coef_)
# print(clf.intercept_)
a = -w[0][0]/w[0][1]
# print(a)
# print(b)
r = np.linspace(0, 1, 1000)
t = a*r + b
b_up_margin = -(clf.intercept_+1)/w[0][1]
b_down_margin = -(clf.intercept_-1)/w[0][1]
up_margin = a*r + b_up_margin
down_margin = a*r + b_down_margin
plt.scatter(normalize_data[:, 0], normalize_data[:, 1])
plt.plot(r, t, color='k')
plt.plot(r, up_margin, color='r')
plt.plot(r, down_margin, color='r')
plt.show()
# print(y_predict)
# print(y_test)



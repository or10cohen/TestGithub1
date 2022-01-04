import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


def split_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    np.random.seed(2)
    np.random.shuffle(X)
    np.random.seed(2)
    np.random.shuffle(y)
    X_train = X[:int(0.7 * len(X)), :]
    X_test = X[int(0.7 * len(X)):, :]
    y_train = y[:int(0.7 * len(X))]
    y_test = y[int(0.7 * len(X)):]
    return X_train, X_test, y_train, y_test


# X_train1, X_test1, y_train1, y_test1 = split_data()
# print(X_train1.shape, X_test1.shape, y_train1.shape, y_test1.shape)


def Distance_Calculation(Training_Data, Test_Point):
    # add_num_of_columns = len(X[0]) - len(Y[0])
    # if add_num_of_columns > 0:
    #     ones_matrix = np.ones((len(Y), add_num_of_columns)) * 10000
    #     Y = np.hstack((Y, ones_matrix))
    # elif add_num_of_columns < 0:
    #     ones_matrix2 = npYte.ones((len(X), add_num_of_columns * (-1))) * 10000
    #     X = np.hstack((len(X), ones_matrix2))

    distance_between_points = Training_Data - Test_Point
    distance_between_points = distance_between_points ** 2
    distance_between_points = np.sum(distance_between_points, axis=1)
    distance_between_points = np.sqrt(np.array(distance_between_points))
    return distance_between_points


def nearest_neighbours(Distance, train_labels, k):
    result = np.argsort(Distance)
    return train_labels[result[:k]]


def Voting(y):
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]


def accuracy(Y_predict, Y_test):
    count = Y_predict == Y_test
    accurec = np.count_nonzero(count)
    return (accurec / len(Y_test)) * 100


# Predict the value in Test

X_train1, X_test1, y_train1, y_test1 = split_data()
All_targets = np.empty((0, len(y_test1)))
for i in range(len(X_test1)):
    distance = Distance_Calculation(X_train1, X_test1[i, :])
    labels = nearest_neighbours(distance, y_train1, 4)
    target = Voting(labels)
    All_targets = np.append(All_targets, target)

print('The accuracy is: {}%'.format(accuracy(All_targets, y_test1)))



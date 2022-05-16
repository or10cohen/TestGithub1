import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
import random
from scipy import spatial
import matplotlib.pyplot as plt


class DBSCAN:
    def __init__(self, Data, number_of_features=3, epsilon=1, minPts=3):
        self.Data = Data
        self.epsilon, self.minPts, self.len_data, self.number_of_features \
            =  epsilon, minPts, len(Data), number_of_features
        self.X = self.data()
        self.normalize_data = self.normalizeData()
        self.distance_matrix = None
        self.count_neighbours = None
        self.cluster = []

    def data(self):
        X = self.Data.data[:, :self.number_of_features]
        return X

    def normalizeData(self):
        normalize_data = MinMaxScaler().fit_transform(self.X_train)
        return normalize_data

    def calcDistancematrix(self, points):
        self.distance_matrix = spatial.distance_matrix(points, self.X_train, p=2)

    def neighboursInEpsilon(self):
        self.distance_matrix = self.distance_matrix <= self.epsilon
        self.count_neighbours = np.count_nonzero(a, axis=1)

    def randomPoint(self):
        random_points = self.X[np.random.randint(self.len_data)]


    def corePoint(self):
        pass

    def makeClusters(self):
        pass

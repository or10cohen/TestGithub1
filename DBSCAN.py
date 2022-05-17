import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
import random
from scipy import spatial
import matplotlib.pyplot as plt


class DBSCAN:
    def __init__(self, data, number_of_features=3, epsilon=1, minPts=3):
        self.Data, self.epsilon, self.minPts, self.len_data, self.number_of_features \
            = data, epsilon, minPts, len(data), number_of_features
        self.X = self.data()
        self.normalize_data = self.normalizeData()
        self.distance_matrix = self.calcDistancematrix()
        self.count_neighbours_in_epsilon = self.neighboursInEpsilon()
        self.list_of_index = list(range(self.len_data))
        self.choose_random_index = None
        self.distance_matrix_true_false = None
        self.distance_matrix_where = None
        self.core_point_true_false = None
        self.cluster = [None] * self.len_data

    def data(self):
        X = self.Data.data[:, :self.number_of_features]
        return X

    def normalizeData(self):
        normalize_data = MinMaxScaler().fit_transform(self.X)
        return normalize_data

    def calcDistancematrix(self):
        distance_matrix = spatial.distance_matrix(self.X, self.X, p=2)
        return distance_matrix

    def randomPoint(self):
        choose_random_index = random.choice(self.list_of_index)
        # self.list_of_index.pop(chose_random_index)
        return choose_random_index

    def neighboursInEpsilon(self):
        self.distance_matrix_true_false = self.distance_matrix <= self.epsilon
        self.distance_matrix_where = np.where(0 < self.distance_matrix <= self.epsilon)
        count_neighbours_in_epsilon = np.count_nonzero(self.distance_matrix, axis=1)
        return count_neighbours_in_epsilon

    def corePoint(self):
        self.core_point_true_false = self.count_neighbours_in_epsilon > self.minPts

    def reachablePoints(self, index):
        reachable_points = self.distance_matrix_true_false[index]
        reachable_points = np.where(reachable_points)
        return reachable_points

    def makeClusters(self, index_of_points, cluster):
        for ins in index_of_points:
            self.cluster[ins] = cluster
        pass

    def run(self):
        #1.choose core point first
        run = True
        while run:
            self.choose_random_index = self.randomPoint()
            if self.core_point_true_false[self.choose_random_index]:
                run = False

        #2.cluster core point + reachable points and pop all
        cluster = 0
        self.list_of_index.pop(self.choose_random_index)
        index_reachable_points = self.reachablePoints(self.choose_random_index)



        self.makeClusters(index_reachable_points, cluster)
        cluster += 1





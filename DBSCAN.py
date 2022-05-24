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
        self.choose_random_index = random.choice(self.list_of_index)
        # # self.list_of_index.pop(chose_random_index)
        # return choose_random_index

    def neighboursInEpsilon(self):
        self.distance_matrix_true_false = self.distance_matrix <= self.epsilon #
        # self.distance_matrix_where = np.where(0 < self.distance_matrix <= self.epsilon) #
        count_neighbours_in_epsilon = np.count_nonzero(self.distance_matrix_true_false, axis=1) #
        return count_neighbours_in_epsilon #

    def corePoint(self):
        self.core_point_true_false = self.count_neighbours_in_epsilon >= self.minPts

    def chooseCorePoint(self):
        run = True
        while run:
            self.randomPoint()
            if self.core_point_true_false[self.choose_random_index]:
                run = False

    def directReachablePoints(self, index):
        direct_reachable_points = self.distance_matrix_true_false[index]
        direct_reachable_points = np.where(direct_reachable_points)
        return direct_reachable_points

    def makeClusters(self, index, cluster):
        for ins in index:
            self.cluster[ins] = cluster

    def popFromIndex(self, index):
        for ins in index:
            self.list_of_index.remove(ins)

    def run(self):
        cluster = 0
        #pop all noise points
        noise_index = np.where(self.count_neighbours_in_epsilon == 0)
        self.popFromIndex(noise_index)
        #cluster all points without noise
        while len(self.list_of_index) > 0:
            self.chooseCorePoint()
            index_reachable_points = self.directReachablePoints(self.choose_random_index)
            self.makeClusters(index_reachable_points, cluster)

            self.popFromIndex(index_reachable_points)

            index_reachable_points = np.delete(index_reachable_points, np.where(index_reachable_points == self.choose_random_index)) #pop corePts
            # index_reachable_points.remove(self.choose_random_index)  #pop corePts

            while len(index_reachable_points) > 0:   # add self.list_of_index.pop
                if self.core_point_true_false[index_reachable_points[0]]:
                    index_reachable_points2 = self.directReachablePoints(index_reachable_points[0])
                    for ind in index_reachable_points2:
                        if self.cluster[ind] is not None:
                            # index_reachable_points2.pop(ind)      #maybe need list
                            index_reachable_points2 = np.delete(index_reachable_points2, np.where(index_reachable_points == idx))
                        self.popFromIndex(ind)
                    index_reachable_points = np.append(index_reachable_points, index_reachable_points2)
                    # index_reachable_points.extend(index_reachable_points2)
                index_reachable_points = np.delete(index_reachable_points, 0)
                # index_reachable_points.pop(0)
                self.makeClusters(index_reachable_points, cluster)
            cluster += 1


if __name__ == '__main__':
    dataset = datasets.load_iris()
    run_DBSCAN = DBSCAN(data=dataset)
    print(run_DBSCAN.len_data)
    print(run_DBSCAN.cluster)
    print(run_DBSCAN.distance_matrix)
    print(run_DBSCAN.distance_matrix_true_false)
    print(run_DBSCAN.count_neighbours_in_epsilon)
    print(run_DBSCAN.core_point_true_false)
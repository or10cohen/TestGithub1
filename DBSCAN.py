import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
import random
from scipy import spatial
import matplotlib.pyplot as plt


class DBSCAN:
    def __init__(self, data, number_of_features=3, epsilon=0.3, minPts=3):
        self.Data, self.epsilon, self.minPts, self.len_data, self.number_of_features \
            = data, epsilon, minPts, len(data.data), number_of_features
        self.X = self.data()
        self.normalize_data = self.normalizeData()
        self.distance_matrix = self.calcDistancematrix()
        self.count_neighbours_in_epsilon, self.distance_matrix_true_false = self.neighboursInEpsilon()
        self.list_of_index = list(range(self.len_data))
        self.choose_random_index = None
        self.distance_matrix_where = None
        self.core_point_true_false = self.corePoint()
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
        distance_matrix_true_false = self.distance_matrix <= self.epsilon
        # self.distance_matrix_where = np.where(0 < self.distance_matrix <= self.epsilon)
        count_neighbours_in_epsilon = np.count_nonzero(distance_matrix_true_false, axis=1)
        return count_neighbours_in_epsilon, distance_matrix_true_false

    def corePoint(self):
        core_point_true_false = self.count_neighbours_in_epsilon >= self.minPts
        return core_point_true_false

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
        noise_index = np.concatenate(np.where(self.count_neighbours_in_epsilon == 1)).tolist() #where and convert to list
        self.popFromIndex(noise_index)
        #cluster all points without noise

        while len(self.list_of_index) > 0:
            self.chooseCorePoint()
            index_reachable_points = self.directReachablePoints(self.choose_random_index)
            index_reachable_points_list = np.concatenate(index_reachable_points).tolist() #convert to list

            self.makeClusters(index_reachable_points_list, cluster)
            self.popFromIndex(index_reachable_points_list)
            index_reachable_points_list.remove(self.choose_random_index) #pop corePts


            while len(index_reachable_points_list) > 0:   # add self.list_of_index.pop
                if self.core_point_true_false[index_reachable_points_list[0]]:
                    index_reachable_points2 = self.directReachablePoints(index_reachable_points_list[0])
                    index_reachable_points2_list = np.concatenate(index_reachable_points2).tolist() #convert to list\
                    index_reachable_points2_list.pop(0)
                    for ind in index_reachable_points2_list:
                        if self.cluster[ind] is not None:
                            index_reachable_points2_list.remove(ind)
                        elif ind in self.list_of_index:
                            self.list_of_index.remove(ind)
                        else:
                            pass
                    index_reachable_points_list.extend(index_reachable_points2_list)
                    # index_reachable_points.extend(index_reachable_points2)
                else:
                    self.makeClusters(index_reachable_points_list[0], cluster)
                    index_reachable_points_list.remove(index_reachable_points_list[0])


                # index_reachable_points_list = np.delete(index_reachable_points_list, 0)
                # index_reachable_points.pop(0)
                self.makeClusters(index_reachable_points_list, cluster)
            cluster += 1


if __name__ == '__main__':
    dataset = datasets.load_iris()
    run_DBSCAN = DBSCAN(data=dataset)
    run_DBSCAN.run()
    # print(run_DBSCAN.len_data)
    # print(run_DBSCAN.distance_matrix)
    # print(run_DBSCAN.distance_matrix_true_false)
    # print(run_DBSCAN.count_neighbours_in_epsilon)
    # print(run_DBSCAN.core_point_true_false)
    print(run_DBSCAN.cluster)

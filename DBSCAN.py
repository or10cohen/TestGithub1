import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
import random
from scipy import spatial
import matplotlib.pyplot as plt
from itertools import compress
from operator import itemgetter


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
    def removeFromIndex(self, index):
        for ins in index:
            self.list_of_index.remove(ins)
    def noisePoints(self):
        noise_index = []
        # print("noise index", noise_index)
        # ----------------------------------------------------# points with neighbours! but without corePts neighbours
        # ---------------------------ליצר ליסט של השכנים ולבדוק אם הם קורפוינטסת אם כל הליסט הוא טרו או פולס. אkkם הכל פולס לשייך אותם לרעשים
        check_point_are_not_corePts = list(compress(self.list_of_index, [not elem for elem in
                                                                         self.core_point_true_false]))  # list index of all not! core point
        # print("check_point_are_not_corePts", check_point_are_not_corePts)
        for point in check_point_are_not_corePts:
            # print('point', point)
            index = [i for i, x in enumerate(self.distance_matrix_true_false[point]) if x]  # index of *nighbares* of any point
            # print("index", index)
            TF_list = itemgetter(*index)(
                self.core_point_true_false)  # check if them T\F [list of true false corePts for any point]
            # print(type(TF_list))
            # print("TF_list", TF_list)
            if TF_list == True:
                TF_list = [True]
            elif TF_list == False:
                TF_list = [False]
            else:
                pass

            if any(TF_list):
                # print(any(TF_list))
                # print("not noise")
                pass
            else:
                # print(any(TF_list))
                # print("noise")
                noise_index.append(point)
            # print(noise_index)
        return noise_index
    def plot_3d(self, cluster_vector, rotate_fig_0 = None, rotate_fig_1 = None):
        x = np.array(self.cluster, dtype=np.float64)
        max_value_in_cluster = np.nanmax(x)
        max_value_in_cluster = int(max_value_in_cluster)
        cluster_tag_x = [[] for i in range(max_value_in_cluster + 1)]
        cluster_tag_y = [[] for i in range(max_value_in_cluster + 1)]
        cluster_tag_z = [[] for i in range(max_value_in_cluster + 1)]
        cluster = 0
        colors = np.array(["red", "green", "blue", "yellow", "pink", "black", "orange", "purple", "beige",\
                           "brown", "cyan", "magenta"])
        fig = plt.figure(figsize=(9, 14))
        ax = fig.add_subplot(2, 1, 1, projection='3d')
        ax.set_title('DBSCAN 3D')
        ax.view_init(rotate_fig_0, rotate_fig_1)
        ax.set_xlabel('Fetcher 1')
        ax.set_ylabel('Fetcher 2')
        ax.set_zlabel('Fetcher 3')
        while cluster <= max_value_in_cluster:
            for idx, i in enumerate(cluster_vector):
                if i == cluster:
                    cluster_tag_x[cluster].append(self.X[idx, 0])
                    cluster_tag_y[cluster].append(self.X[idx, 1])
                    cluster_tag_z[cluster].append(self.X[idx, 2])
                elif i == None:
                    ax.scatter(self.X[idx, 0], self.X[idx, 1], self.X[idx, 2], marker='x', c="gray")
            ax.scatter3D(cluster_tag_x[cluster], cluster_tag_y[cluster], cluster_tag_z[cluster], c=colors[cluster])
            cluster += 1
        plt.savefig('C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\DBSACN_3D.png')
        # plt.show()
    def plot_2d(self, cluster_vector):
        x = np.array(self.cluster, dtype=np.float64)
        max_value_in_cluster = np.nanmax(x)
        max_value_in_cluster = int(max_value_in_cluster)
        cluster_tag_x = [[] for i in range(max_value_in_cluster + 1)]
        cluster_tag_y = [[] for i in range(max_value_in_cluster + 1)]
        cluster = 0
        colors = np.array(["red", "green", "blue", "yellow", "pink", "black", "orange", "purple", "beige", \
                            "brown", "cyan", "magenta"])
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        ax.set_title('DBSCAN 2D')
        while cluster <= max_value_in_cluster:
            for idx, i in enumerate(cluster_vector):
                if i == cluster:
                    cluster_tag_x[cluster].append(self.X[idx, 0])
                    cluster_tag_y[cluster].append(self.X[idx, 1])
                elif i == None:
                    ax.scatter(self.X[idx, 0], self.X[idx, 1], marker='x', c="gray")

            ax.scatter(cluster_tag_x[cluster], cluster_tag_y[cluster], c=colors[cluster])
            cluster += 1

        plt.savefig('C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\DBSACN_2D.png')
        # plt.show()
    def run(self):
        cluster = 0
        noise_index = self.noisePoints()
        self.removeFromIndex(noise_index)
        # print("\n\n\n-------------------------------------------------------------\n\n\n")
        while len(self.list_of_index) > 0:
            self.chooseCorePoint()
            # print("core point", self.choose_random_index)
            index_reachable_points = self.directReachablePoints(self.choose_random_index)
            index_reachable_points_list = index_reachable_points[0].tolist() #convert to list
            # print('index_reachable_points_list', index_reachable_points_list)
            # print('len index_reachable_points_list', len(index_reachable_points_list))
            # print("\n\n\n-------------------------------------------------------------\n\n\n")
            # while len(index_reachable_points_list) > 0:   # add self.list_of_index.pop
            for point in index_reachable_points_list:
                # print("index_reachable_points_list", index_reachable_points_list)
                # print("point", point)
                if self.core_point_true_false[point]:
                    # print("T\F", self.core_point_true_false[point])
                    index_reachable_points2 = self.directReachablePoints(point)
                    index_reachable_points2_list = index_reachable_points2[0].tolist() #convert to list\
                    # print('index_reachable_points2_list', index_reachable_points2_list)
                    for point2 in index_reachable_points2_list:
                        # print("i", point2)
                        if point2 in index_reachable_points_list:
                            # print("not need you!")
                            pass
                        else:
                            # print("I need you!")
                            index_reachable_points_list.append(point2)
                else:
                    # print("T\F", self.core_point_true_false[point])
                    pass

            self.makeClusters(index_reachable_points_list, cluster)
            self.removeFromIndex(index_reachable_points_list)
            # print("len(self.list_of_index)", len(self.list_of_index))
            index_reachable_points = []
            cluster += 1
            # print("cluster += 1, cluster:", cluster)
            # print("\n\n\n-------------------------------------------------------------\n\n\n")



if __name__ == '__main__':
    dataset = datasets.load_iris()
    run_DBSCAN = DBSCAN(data=dataset, number_of_features=3, epsilon=0.3, minPts=3)
    run_DBSCAN.run()
    run_DBSCAN.plot_3d(run_DBSCAN.cluster)
    run_DBSCAN.plot_2d(run_DBSCAN.cluster)
    # print("cluster per index", run_DBSCAN.cluster)
    # print("plot x", run_DBSCAN.plot_3d(run_DBSCAN.cluster))
    circle = datasets.make_circles(n_samples=100, shuffle=True, noise=None, random_state=None, factor=0.4)
    blobs = datasets.make_s_curve(n_samples=100, noise=0.2, random_state=None)
    print(type(circle[0]))
    print(type(blobs[0]))
    print(circle[0].shape)
    print(blobs[0])
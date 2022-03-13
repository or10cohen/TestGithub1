import numpy as np
from sklearn import datasets
from scipy import spatial
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt


dataset = datasets.load_iris()
X = dataset.data[:, :]

class HierarchicalClustering:

    def __init__(self, Data, number_clusters=2, max_distance=200,  linkage_method='complete'):
        self.train_data = Data
        self.number_clusters = number_clusters
        self.max_distance = max_distance
        self.linkage_method = linkage_method
        self.len_train_data = len(Data)
        self.Normalize_training_data = self.normalize_data()
        self.distance_matrix = self.calc_distance_matrix()
        self.clusters = [[i] for i in range(len(self.train_data))]

    def normalize_data(self):
        Normalize_training_data = np.copy(self.train_data)
        for i in range(Normalize_training_data.shape[1]):
            for j in range(Normalize_training_data.shape[0]):
                Normalize_training_data[j,i] = (Normalize_training_data[j,i] - np.ndarray.min(Normalize_training_data[:,i])) / \
                    (np.ndarray.max(Normalize_training_data[:,i]) - np.ndarray.min(Normalize_training_data[:,i]))
        return Normalize_training_data

    def calc_distance_matrix(self):
        distance_matrix = spatial.distance_matrix(self.Normalize_training_data, self.Normalize_training_data, p=2)
        distance_matrix += np.diag([np.inf] * len(self.train_data))
        return distance_matrix

    def updateDistanceMatrix(self, index1):
        self.distance_matrix = np.delete(self.distance_matrix, index1[1], axis=0)
        first_column = self.distance_matrix[:, index1[1]]
        second_column = self.distance_matrix[:, index1[0]]
        if self.linkage_method == 'single':
             new_column = np.minimum(first_column, second_column)
        elif self.linkage_method == 'complete':
             new_column = np.maximum(first_column, second_column)
        elif self.linkage_method == 'average':
             new_column = (np.maximum(first_column, second_column) + np.minimum(first_column, second_column)) / 2
        else:
            pass
        new_row = new_column.T
        self.distance_matrix = np.delete(self.distance_matrix, index1[1], axis=1)
        self.distance_matrix[:, index1[0]] = new_column
        self.distance_matrix[index1[0], :] = new_row    # its new row here, but we can input column and it automaticly did Transfor.
        self.distance_matrix += np.diag([np.inf] * len(self.distance_matrix))

    # update clusters
    def updateCluster(self, index1):
        self.clusters[index1[0]].extend(self.clusters[index1[1]])
        self.clusters.pop(index1[1])

    def fit(self):
        while len(self.clusters) > self.number_clusters and np.min(self.distance_matrix) < self.max_distance:
            index1, index2 = np.where(self.distance_matrix == np.min(self.distance_matrix))
            self.updateCluster(index1)
            self.updateDistanceMatrix(index1)

    def print_3d(self,rotate_fig_0 = None, rotate_fig_1 = None):
        if len(self.clusters) > self.number_clusters:
            len_we_need = len(self.clusters)
        else:
            len_we_need = self.number_clusters
        self.x_label = [np.zeros(self.len_train_data) for i in range(self.number_clusters)]
        self.y_label = [np.zeros(self.len_train_data) for i in range(self.number_clusters)]
        self.z_label = [np.zeros(self.len_train_data) for i in range(self.number_clusters)]
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        for i in len_we_need:
            for j in self.clusters[i]:
                self.x_label[i][j] = self.train_data[j, 0]
                self.y_label[i][j] = self.train_data[j, 1]
                self.z_label[i][j] = self.train_data[j, 2]
            ax.scatter3D(self.x_label[i][self.x_label[i] != 0], self.y_label[i][self.y_label[i] != 0], self.z_label[i][self.z_label[i] != 0])
        ax.set_title('My_linkage')
        ax.view_init(rotate_fig_0, rotate_fig_1)

        sklearn_linkage = AgglomerativeClustering(n_clusters=len_we_need, linkage=self.linkage_method)
        sklearn_linkage.fit(self.train_data)
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.scatter3D(self.train_data[:, 0], self.train_data[:, 1], self.train_data[:, 2], c=sklearn_linkage.labels_)
        ax.set_title('sklearn_linkage AgglomerativeClustering')
        ax.view_init(rotate_fig_0, rotate_fig_1)
        # plt.show()
        plt.savefig('C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\Print_3d.png')

    def print_2d(self):
        if len(self.clusters) > self.number_clusters:
            len_we_need = len(self.clusters)
        else:
            len_we_need = self.number_clusters
        self.x_label = [np.zeros(self.len_train_data) for i in range(len_we_need)]
        self.y_label = [np.zeros(self.len_train_data) for i in range(len_we_need)]
        plt.figure(figsize=(8, 4))
        plt.subplot(121)
        plt.title('My_linkage')
        for i in range(len_we_need):
            for j in self.clusters[i]:
                self.x_label[i][j] = self.train_data[j, 0]
                self.y_label[i][j] = self.train_data[j, 1]
            plt.scatter(self.x_label[i][self.x_label[i]!=0], self.y_label[i][self.y_label[i]!=0])

        sklearn_linkage = AgglomerativeClustering(n_clusters=len_we_need, linkage=self.linkage_method)  ##changer here!!! len(self.clusters) <<--->> self.number_clusters
        sklearn_linkage.fit(self.train_data)
        plt.subplot(122)
        plt.title('sklearn_linkage AgglomerativeClustering')
        plt.scatter(self.train_data[:, 0], self.train_data[:, 1], c=sklearn_linkage.labels_)
        # plt.show()
        plt.savefig('C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\Print_2d.png')


# if __name__ = 'main':

#
HC = HierarchicalClustering(X)
HC.fit()


import numpy as np
import scipy
from sklearn import datasets
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import streamlit as st



dataset = datasets.load_iris()
X = dataset.data[:, :]

class HierarchicalClustering:

    def __init__(self, Data):
        self.train_data = Data
        self.Normalize_training_data = self.Normalize_data()
        self.distance_matrix = self.Calc_distance_matrix()
        self.clusters = [[i] for i in range(len(self.train_data))]

    def Normalize_data(self):
        Normalize_training_data = np.copy(self.train_data)
        for i in range(Normalize_training_data.shape[1]):
            for j in range(Normalize_training_data.shape[0]):
                Normalize_training_data[j,i] = (Normalize_training_data[j,i] - np.ndarray.min(Normalize_training_data[:,i])) / \
                    (np.ndarray.max(Normalize_training_data[:,i]) - np.ndarray.min(Normalize_training_data[:,i]))
        return Normalize_training_data

    def Calc_distance_matrix(self):
        distance_matrix = scipy.spatial.distance_matrix(self.Normalize_training_data, self.Normalize_training_data, p=2)
        distance_matrix += np.diag([np.inf] * len(self.train_data))
        return distance_matrix

    def UpdateDistanceMatrix(self, index1):
        self.distance_matrix = np.delete(self.distance_matrix, index1[1], axis=0)
        first_column = self.distance_matrix[:, index1[1]]
        second_column = self.distance_matrix[:, index1[0]]
        # if 'minimun':
        #     # new_column = np.minimum(first_column, second_column)
        # if 'maximum':
        #     new_column = np.maximum(first_column, second_column)
        # if 'avarage':
        #     new_column = (np.maximum(first_column, second_column) + np.minimum(first_column, second_column)) / 2
        new_column = np.maximum(first_column, second_column)
        # new_row = np.reshape(new_column, (1, len(new_column)))
        self.distance_matrix = np.delete(self.distance_matrix, index1[1], axis=1)
        self.distance_matrix[:, index1[0]] = new_column
        self.distance_matrix[index1[0], :] = new_column    # its new row here, but we can input column and it automaticly did Transfor.
        self.distance_matrix += np.diag([np.inf] * len(self.distance_matrix))

    # update clusters
    def UpdateCluster(self, index1):
        self.clusters[index1[0]].extend(self.clusters[index1[1]])
        self.clusters.pop(index1[1])

    def fit(self, number_clusters=2):
        while len(self.clusters) > number_clusters:
            index1, index2 = np.where(self.distance_matrix == np.min(self.distance_matrix))
            self.UpdateCluster(index1)
            self.UpdateDistanceMatrix(index1)

    def Print_3d(self, number_clusters=2):
        HC = HierarchicalClustering(X)
        HC.fit(number_clusters)
        self.x_label = [np.zeros(len(X)) for i in range(number_clusters)]
        self.y_label = [np.zeros(len(X)) for i in range(number_clusters)]
        self.z_label = [np.zeros(len(X)) for i in range(number_clusters)]
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        for i in range(number_clusters):
            for j in HC.clusters[i]:
                self.x_label[i][j] = X[j, 0]
                self.y_label[i][j] = X[j, 1]
                self.z_label[i][j] = X[j, 2]
            ax.scatter(self.x_label[i][self.x_label[i] != 0], self.y_label[i][self.y_label[i] != 0], self.z_label[i][self.z_label[i] != 0])
        ax.set_title('My_linkage')

        sklearn_linkage = AgglomerativeClustering(n_clusters=number_clusters, linkage='complete')
        sklearn_linkage.fit(X)
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=sklearn_linkage.labels_)
        ax.set_title('sklearn_linkage AgglomerativeClustering')
        plt.show()

    def Print_2d(self, number_clusters=2):
        HC = HierarchicalClustering(X)
        HC.fit(number_clusters)
        self.x_label = [np.zeros(len(X)) for i in range(number_clusters)]
        self.y_label = [np.zeros(len(X)) for i in range(number_clusters)]
        plt.figure()
        plt.subplot(121)
        plt.title('My_linkage')
        for i in range(number_clusters):
            for j in HC.clusters[i]:
                self.x_label[i][j] = X[j, 0]
                self.y_label[i][j] = X[j, 1]
            plt.scatter(self.x_label[i][self.x_label[i]!=0], self.y_label[i][self.y_label[i]!=0])

        sklearn_linkage = AgglomerativeClustering(2, linkage='complete')
        sklearn_linkage.fit(X)
        plt.subplot(122)
        plt.title('sklearn_linkage AgglomerativeClustering')
        plt.scatter(X[:, 0], X[:, 1], c=sklearn_linkage.labels_)
        plt.show()


HC = HierarchicalClustering(X)
# HC.Print_2d()
# HC.Print_3d()


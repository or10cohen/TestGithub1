import numpy as np
import scipy
from sklearn import datasets
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data[:, :]

class HierarchicalClustering:

    def __init__(self, Data):
        self.train_data = Data
        self.Normalize_training_data = self.Normalize_data()
        self.distance_matrix = self.Calc_distance_matrix()
        self.clusters = [[i] for i in range(len(self.train_data))]

    def Normalize_data(self):
        self.Normalize_training_data = np.copy(self.train_data)
        for i in range(self.Normalize_training_data.shape[1]):
            for j in range(self.Normalize_training_data.shape[0]):
                self.Normalize_training_data[j,i] = (self.Normalize_training_data[j,i] - np.ndarray.min(self.Normalize_training_data[:,i])) / \
                    (np.ndarray.max(self.Normalize_training_data[:,i]) - np.ndarray.min(self.Normalize_training_data[:,i]))
        return self.Normalize_training_data

    def Calc_distance_matrix(self):
        self.distance_matrix = scipy.spatial.distance_matrix(self.Normalize_training_data, self.Normalize_training_data, p=2)
        self.distance_matrix += np.diag([np.inf] * len(self.train_data))
        return self.distance_matrix

    def UpdateDistanceMatrix(self, index1):
        self.distance_matrix = np.delete(self.distance_matrix, index1[1], axis=0)
        first_column = self.distance_matrix[:, index1[1]]
        second_column = self.distance_matrix[:, index1[0]]
        new_column = np.maximum(first_column, second_column)
        # new_row = np.reshape(new_column, (1, len(new_column)))
        self.distance_matrix = np.delete(self.distance_matrix, index1[1], axis=1)
        self.distance_matrix[:, index1[0]] = new_column
        self.distance_matrix[index1[0], :] = new_column
        self.distance_matrix += np.diag([np.inf] * len(self.distance_matrix))

    # update clusters
    def UpdateCluster(self, index1):
        self.clusters[index1[0]].extend(self.clusters[index1[1]])
        self.clusters.pop(index1[1])

    def fit(self, number_clusters=2):
        while len(self.clusters) > number_clusters:
            # print(len(self.clusters))
            index1, index2 = np.where(self.distance_matrix == np.min(self.distance_matrix))
            # print(index1)
            self.UpdateCluster(index1)
            self.UpdateDistanceMatrix(index1)


HC = HierarchicalClustering(X)
HC.fit()
plt.figure()
plt.subplot(121)
plt.title('My_linkage')
x_label1 = np.zeros(150)
y_label1 = np.zeros(150)
for i in HC.clusters[0]:
    x_label1[i] = X[i ,0]
    y_label1[i] = X[i ,1]
x_label2 = np.zeros(150)
y_label2 = np.zeros(150)
for i in HC.clusters[1]:
    x_label2[i] = X[i ,0]
    y_label2[i] = X[i ,1]
plt.scatter(x_label1[x_label1!=0], y_label1[y_label1!=0])
plt.scatter(x_label2[x_label2!=0], y_label2[y_label2!=0])

sklearn_linkage = AgglomerativeClustering(2, linkage='complete')
sklearn_linkage.fit(X)
plt.subplot(122)
plt.title('sklearn_linkage AgglomerativeClustering')
plt.scatter(X[:,0], X[:,1], c = sklearn_linkage.labels_)
plt.show()



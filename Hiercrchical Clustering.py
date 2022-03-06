import numpy as np
import scipy
from sklearn import datasets
from scipy.spatial.distance import squareform, pdist

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
print(len(HC.clusters))
print(HC.clusters[0])
print(len(HC.clusters[0]))
print(HC.clusters[1])
print(len(HC.clusters[1]))




# sns.heatmap(HC.distance_matrix[:10, :10], cmap=plt.cm.Reds)
# plt.show()
# i = np.where(HC.distance_matrix == np.min(HC.distance_matrix))
# print(i[0][0])

import matplotlib.pyplot
import numpy as np
import scipy
import sklearn
from matplotlib import pyplot as plt
from sklearn import datasets
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from scipy.spatial import distance_matrix
import seaborn as sns
import scipy.cluster.hierarchy as sch

# from skbio.stats.distance import DissimilarityMatrix


iris = datasets.load_iris()
X = iris.data[:, :]

# normalize_data = sklearn.preprocessing.normalize(X, norm='l2', axis=0, copy=True)
# new_data = pd.DataFrame(X)
# new_data.index += 1
# dis_matrix = pd.DataFrame(squareform(pdist(new_data.loc[:, :])), columns=new_data.index, index=new_data.index)


class HierarchicalClustering:

    def __init__(self, Data):
        self.train_data = Data
        # self.Normalize_training_data = self.Normalize_data()
        self.distance_matrix = self.Calc_distance_matrix()
        self.clusters = [[i] for i in range(len(self.train_data))]

    # def Normalize_data(self):
    #     self.Normalize_training_data = np.linalg.norm(self.train_data)
    #     return self.Normalize_training_data

    def Calc_distance_matrix(self):
        self.distance_matrix = scipy.spatial.distance_matrix(self.train_data \
                                                             , self.train_data, p=2)
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

# sns.heatmap(HC.distance_matrix[:10, :10], cmap=plt.cm.Reds)
# plt.show()
# i = np.where(HC.distance_matrix == np.min(HC.distance_matrix))
# print(i[0][0])

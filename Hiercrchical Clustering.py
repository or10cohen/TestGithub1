
import numpy as np
import scipy
import sklearn
import pandas as pd
from sklearn import datasets
from scipy.spatial import distance_matrix

import scipy.cluster.hierarchy as sch

import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset = True)

# from skbio.stats.distance import DissimilarityMatrix


iris = datasets.load_iris()
iris_data = iris.data[:, :]


class HierarchicalClustering:

    def __init__(self, Data):
        self.train_data = Data
        self.Normalize_training_data = self.Normalize_data()
        self.distance_matrix = self.Calc_distance_matrix()
        self.lst = [[] for i in range(3)]
        self.Min_value_in_distance_matrix= self.Min_value_in_distance_matrix()
    def Normalize_data(self):
        Normalize_training_data = sklearn.preprocessing.normalize(self.train_data, norm='l2', axis=0, copy=True)
        return Normalize_training_data

    def Calc_distance_matrix(self):
        self.distance_matrix = scipy.spatial.distance_matrix(self.Normalize_training_data \
                                                             , self.Normalize_training_data, p=2)
        self.distance_matrix = pd.DataFrame(self.distance_matrix)
        return self.distance_matrix

    def Min_value_in_distance_matrix(self):
        distance_matrix_without_zero = self.distance_matrix.replace(0, None)
        Min_value_in_distance_matrix = distance_matrix_without_zero.min().min()
        return Min_value_in_distance_matrix

    def UpdateDistanceMatrix(self, index1, index2):
        self.distance_matrix = np.delete(self.distance_matrix, (index1[0], index2[0]), axis=0)
        first_column = self.distance_matrix[:, index1[1]]
        second_column = self.distance_matrix[:, index2[1]]
        max_column = np.maximum(first_column, second_column)
        self.distance_matrix = np.delete(self.distance_matrix, (index1[1], index2[1]), axis=1)





HC = HierarchicalClustering(iris_data)

# import seaborn as sns
# from matplotlib import pyplot as plt
# sns.heatmap(HC.distance_matrix[:10, :10], cmap=plt.cm.Reds)
# plt.show()


# i, j = np.where(HC.distance_matrix == np.min(HC.distance_matrix[np.nonzero(HC.distance_matrix)]))
# print(i[0], j[0])


print(Fore.LIGHTGREEN_EX + '\nshape train data:', iris_data.shape)
print(Fore.LIGHTBLUE_EX + '\nNormalize training data sample:\n', HC.Normalize_training_data[:5, :])
print(Fore.LIGHTBLUE_EX + '\ntype distance Matrix:', type(HC.distance_matrix))
print(Fore.LIGHTBLUE_EX + '\nshape distance Matrix:', HC.distance_matrix.shape)
print(Fore.LIGHTBLUE_EX + '\ndistance Matrix:\n', HC.distance_matrix)
print(Fore.LIGHTBLUE_EX + '\nMin_value_in_distance_matrix\n', HC.Min_value_in_distance_matrix)

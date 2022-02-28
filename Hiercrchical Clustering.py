import matplotlib.pyplot
import numpy as np
import scipy
import sklearn
from matplotlib import pyplot as plt
from sklearn import datasets
from scipy.spatial import distance_matrix
import seaborn as sns
import scipy.cluster.hierarchy as sch

# from skbio.stats.distance import DissimilarityMatrix


iris = datasets.load_iris()
X = iris.data[:, :]


class HierarchicalClustering:

    def __init__(self, Data):
        self.train_data = Data
        self.Normalize_training_data = self.Normalize_data()
        self.distance_matrix = self.Calc_distance_matrix()
        self.lst = [[] for i in range(3)]

    def Normalize_data(self):
        self.Normalize_training_data = sklearn.preprocessing.normalize(self.train_data, norm='l2', axis=0, copy=True)
        return self.Normalize_training_data

    def Calc_distance_matrix(self):
        self.distance_matrix = scipy.spatial.distance_matrix(self.Normalize_training_data \
                                                             , self.Normalize_training_data, p=2)

        return self.distance_matrix

    def UpdateDistanceMatrix(self, index1, index2):
        self.distance_matrix = np.delete(self.distance_matrix, (index1[0], index2[0]), axis=0)
        first_column = self.distance_matrix[:, index1[1]]
        second_column = self.distance_matrix[:, index2[1]]
        max_column = np.maximum(first_column, second_column)
        self.distance_matrix = np.delete(self.distance_matrix, (index1[1], index2[1]), axis=1)



print(Fore.LIGHTGREEN_EX + '\nshape train data:', X.shape)
print(Fore.LIGHTBLUE_EX + '\nNormalize training data sample:\n', HC.Normalize_training_data[:5, :])
print(Fore.LIGHTBLUE_EX + '\ntype distance Matrix:', type(HC.distance_matrix))
print(Fore.LIGHTBLUE_EX + '\nshape distance Matrix:', HC.distance_matrix.shape)
print(Fore.LIGHTBLUE_EX + '\ndistance Matrix sample:\n', HC.distance_matrix[:5, :5])



HC = HierarchicalClustering(X)
# sns.heatmap(HC.distance_matrix[:10, :10], cmap=plt.cm.Reds)
# plt.show()
i, j = np.where(HC.distance_matrix == np.min(HC.distance_matrix[np.nonzero(HC.distance_matrix)]))
print(i[0], j[0])



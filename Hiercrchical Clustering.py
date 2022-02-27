import colorama
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style
colorama.init(autoreset = True)
import numpy as np
import scipy
import sklearn
from matplotlib import pyplot as plt
from sklearn import datasets
from scipy.spatial import distance_matrix
import seaborn as sns
# from skbio.stats.distance import DissimilarityMatrix

iris = datasets.load_iris()
X = iris.data[:, :]
print(Fore.LIGHTGREEN_EX + '\nshape train data:', X.shape)


class HierarchicalClustering:

    def __init__(self, Data):
        self.train_data = Data
        self.Normalize_training_data = self.Normalize_data()
        self.distance_matrix = self.Calc_distance_matrix()

    def Normalize_data(self):
        self.Normalize_training_data = sklearn.preprocessing.normalize(self.train_data, norm='l2', axis=0, copy=True)
        return self.Normalize_training_data

    def Calc_distance_matrix(self):
        self.distance_matrix = scipy.spatial.distance_matrix(self.Normalize_training_data \
                                          , self.Normalize_training_data, p=2)
        return self.distance_matrix

    # def


HC = HierarchicalClustering(X)

print(Fore.LIGHTBLUE_EX + '\nNormalize training data sample:\n', HC.Normalize_training_data[:5, :])
print(Fore.LIGHTBLUE_EX + '\ntype distance Matrix:', type(HC.distance_matrix))
print(Fore.LIGHTBLUE_EX + '\nshape distance Matrix:', HC.distance_matrix.shape)
print(Fore.LIGHTBLUE_EX + '\ndistance Matrix sample:\n', HC.distance_matrix[:5, :5])
# sns.heatmap(HC.distance_matrix[:10, :10], cmap=plt.cm.Reds)
# plt.show()
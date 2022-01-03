import numpy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset = True)

iris = load_iris()     # iris have 3 list. data, target, target name.

# print('the iris type is:', type(iris))
# print('iris data', iris)

print(Fore.RED + '\nthe first array in the iris dictionary is the data. iris.data[1:10]  (note: show less) : \n', iris.data[1:10])
print(Fore.RED + '\nthe second array in the iris dictionary is the target. iris.target:\n', iris.target)
print(Fore.RED + '\nthe third array in the iris dictionary is the names. iris.target_names:\n', iris.target_names)

def DistanceFunction(PointData, PointTest):
    print('print PointData: \n',PointData[:10])
    print('print PointTest: \n',PointTest[:10])
    Distance = PointData - PointTest                                     ## matrix - vector = every line in matrix - vector
    print('print PointData - PointTest: \n', Distance[:10])
    Distance = Distance ** 2
    print('print Distance ** 2: \n',Distance[:10])
    Distance = np.sum(Distance, axis=1).tolist()
    print('print np.sum(Distance, axis=1).tolist(): \n', Distance[:10])
    Distance = np.power(Distance, 0.5)
    print('print Distance ** 0.5: \n', Distance[:10])
    return Distance

# PointData = np.array([[3, 3, 3], [2, 2, 2], [1, 1, 1]])
PointTest = np.array([1, 1, 1, 1])
print(Fore.RED + '\nreturn Distance:\n', DistanceFunction(np.array(iris.data), PointTest))


# def KNearestNeighbours(K, Point):
#     ListKNearestNeighbours = []
#     for i in iris.data:
#         xData, yData, zData, wData = i[0], i[1], i[2], i[3]
#         ListKNearestNeighbours.append(DistanceFunction(xData, xTest, yData, yTest, zData, zTest, wData, wTest))
#     ListKNearestNeighboursSort = ListKNearestNeighbours.copy()
#     ListKNearestNeighboursSort.sort()
#     ListKNearestNeighboursSort = ListKNearestNeighboursSort[:K]
#     return ListKNearestNeighboursSort, ListKNearestNeighbours, K
#
# ListKNearestNeighboursSort, ListKNearestNeighbours, K = KNearestNeighbours(6, 1, 2, 3, 4)
# print('We have list in size K: {} are describe the closet distance points between the Data points and Test point: \n{}'
# .format(K, ListKNearestNeighboursSort))
#
# # print(ListKNearestNeighbours.index(ListKNearestNeighboursSort[0]))
#
# labels = len(np.unique(iris.target))
# def VoteLabelFunction():
#     ListVote = []
#     NumListVote = []
#     for i in range(K):
#         ListVote.append(ListKNearestNeighbours.index(ListKNearestNeighboursSort[i]))
#         ListVote[i] = iris.target[ListVote[i]]
#     for label in range(labels):
#         NumListVote.append(ListVote.count(label))
#     return ListVote ,NumListVote
#
#
# ListVote ,NumListVote = VoteLabelFunction()
# print('\nthe count of label {}: {} \nthe count of label {}: {} \nthe count of label {}: {}' .format(labels - 3, NumListVote[0], labels - 2, NumListVote[1], labels - 1, NumListVote[2]))
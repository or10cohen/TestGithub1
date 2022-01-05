import numpy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset = True)

DataBase = load_iris()     # iris have 3 list. data, target, target name.

# print(Fore.RED + '\nthe first array in the iris dictionary is the data. iris.data[1:10]  (note: show less) : \n', DataBase.data[1:10])
# print(Fore.RED + '\nthe second array in the iris dictionary is the target. iris.target:\n', DataBase.target)
# print(Fore.RED + '\nthe third array in the iris dictionary is the names. iris.target_names:\n', DataBase.target_names)

def SplitData():
    DataPoints = DataBase.data
    TargetPoints = DataBase.target
    RandNumober = np.random.randint(10)
    np.random.seed(RandNumober)
    np.random.shuffle(DataPoints)
    np.random.seed(RandNumober)
    np.random.shuffle(TargetPoints)
    TrainDataPoints = DataPoints[:int(0.7 * len(DataPoints)), :]
    TestDataPoints = DataPoints[int(0.7 * len(DataPoints)):, :]
    TrainTargetPoints = TargetPoints[:int(0.7 * len(TargetPoints))]
    TestTargetPoints = TargetPoints[int(0.7 * len(TargetPoints)):]
    return TrainDataPoints, TestDataPoints, TrainTargetPoints, TestTargetPoints

TrainDataPoints, TestDataPoints, TrainTargetPoints, TestTargetPoints = SplitData()

def DistanceFunction(DataPoints, TestPoint):
    Distance = DataPoints - TestPoint                                  ## matrix - vector = every line in matrix - vector
    Distance = Distance ** 2
    Distance = np.sum(Distance, axis=1).tolist()
    Distance = np.power(Distance, 0.5)                                   ## we can use np.sqrt(nDistance)
    return Distance

PointTest = np.array([1, 1, 1, 1])
print(Fore.RED + '\nreturn Distance:\n', DistanceFunction(TrainDataPoints, PointTest))


# def NearestNeighbours(K, Point):
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
import numpy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset = True)


iris = load_iris()     # iris is a dictionary that have 3 list. data, target, target name.

# print('iris data', iris)
print(Fore.RED + '\nthe first array in the iris dictionary is the data. iris.data[1:10]  (note: show less) : \n', iris.data[1:10])
print(Fore.RED + '\nthe second array in the iris dictionary is the target. iris.target:\n', iris.target)
print(Fore.RED + '\nthe third array in the iris dictionary is the names. iris.target_names:\n', iris.target_names)


print('\n\n')

def DistanceFunction(xData, yData, xTest, yTest):
    Distance =  ((xTest - xData) ** 2 + (yTest - yData) ** 2) ** (0.5)
    return Distance
# print(Fore.RED + '\nreturn Distance:\n', DistanceFunction(0, 2, 0, 3))

def KNearestNeighbours(K, xTest, yTest):
    ListKNearestNeighbours = []
    for i in iris.data:
        xData = i[0]
        yData = i[1]
        ListKNearestNeighbours.append(DistanceFunction(xData, yData, xTest, yTest))
    ListKNearestNeighbours = numpy.array(ListKNearestNeighbours)
    ListKNearestNeighbours.sort()
    # print(ListKNearestNeighbours.index(ListKNearestNeighbours[2]))
    ListKNearestNeighbours = ListKNearestNeighbours[:K]
    return ListKNearestNeighbours


print(KNearestNeighbours(5, 0, 0))




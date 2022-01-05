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
    TrainDataPoints = DataPoints[:int(0.7 * len(DataPoints)), :]                        #DataPoints is aMatrix[:, :]
    TestDataPoints = DataPoints[int(0.7 * len(DataPoints)):, :]                         #DataPoints is aMatrix[:, :]
    TrainTargetPoints = TargetPoints[:int(0.7 * len(TargetPoints))]                     #TargetPoints is a Vector[:]
    TestTargetPoints = TargetPoints[int(0.7 * len(TargetPoints)):]                      #TargetPoints is a Vector[:]
    return TrainDataPoints, TestDataPoints, TrainTargetPoints, TestTargetPoints

# TrainDataPoints, TestDataPoints, TrainTargetPoints, TestTargetPoints = SplitData()

def DistanceFunction(DataPoints, CheckPoint):
    Distance = DataPoints - CheckPoint                                  ## matrix - vector = every line in matrix - vector
    Distance = Distance ** 2
    Distance = np.sum(Distance, axis=1).tolist()
    Distance = np.power(Distance, 0.5)                                   ## we can use np.sqrt(nDistance)
    return Distance

# CheckPoint = np.array([2, 1, 3, 3])
# DistanceVector = DistanceFunction(TrainDataPoints, CheckPoint)

def LabelsNearestNeighbours(DistanceVector, TrainTargetPoints, K):
    NearestNeighboursLabel = np.argsort(DistanceVector) #  example! x = np.array([2,1,3,4,4,0]) ; y = np.array([0,1,2,3,4,5]) ; print(y[np.argsort(x)[:2]]) ==== [5 1]
    return TrainTargetPoints[NearestNeighboursLabel[:K]]

# MyLabelFromNearestNeighbours = LabelsNearestNeighbours(DistanceVector, TrainTargetPoints, 12)

def Vote(MyLabelFromNearestNeighbours):
    values, count = np.unique(MyLabelFromNearestNeighbours, return_counts=True) # print array of unique *values* and array *count* from any unique values
    return values[np.argmax(count)] # example values, count = np.unique(np.array([0,0,1,1,1,2]), return_counts=True) ; values = [0 1 2] count = [2 3 1]

# MaxVoteLabel = Vote(MyLabelFromNearestNeighbours)

def Accuracy(Y_predict, Y_test):
    count = Y_predict == Y_test
    accurec = np.count_nonzero(count)
    return (accurec / len(Y_test)) * 100

TrainDataPoints, TestDataPoints, TrainTargetPoints, TestTargetPoints = SplitData()
All_targets = np.empty((0, len(TestTargetPoints)))
for i in range(len(TestDataPoints)):
    distance = DistanceFunction(TrainDataPoints, TestDataPoints[i, :])
    labels = LabelsNearestNeighbours(distance, TrainTargetPoints, 7)
    target = Vote(labels)
    All_targets = np.append(All_targets, target)

print('The accuracy is: {}%'.format(Accuracy(All_targets, TestTargetPoints)))



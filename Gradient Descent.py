import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 1, 2])
x = np.array([0, 1, 2])
y = np.array([1, 3, 7])

iter = 200
learningRate = [0.01, 0.1, 1]

def GradientDecsent(x, y, learningRate = 1, iter = 50):
    m = 0
    b = 0
    for _ in range(iter):
        for i in range(len(x)):
            xi = x[i]
            yi = y[i]
            guess = m * xi + b
            error = yi - guess
            m = m + error * xi * learningRate
            b = b + error * learningRate
    return m, b


GradientDecsent(x, y)
# print(m)
# print(b)
# guss = y1
#
# for i in range(len(x))
#     CostFunction = (guss[i] - y[i]) ** 2
#     CostFunction += CostFunction

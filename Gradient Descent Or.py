import numpy as np
import matplotlib.pyplot as plt

LearningRates = [1, 0.1, 0.01]
x = np.array([0, 1 ,2], dtype='int64')
y = np.array([1, 3, 7], dtype='int64')
iter = 100

AllLossFunction = []
AllLossFunctionPerLearningRate = []

for LearningRate in LearningRates:
    a, b, c = 2, 2, 0
    for _ in range(iter):
        LossFunction = np.float64(sum((y - (a + b * x + c * (x ** 2))) ** 2))
        AllLossFunction.append(LossFunction)

        D_LossFunction_D_a = 2 * (y - (a + b * x + c * (x ** 2))) * (-1)
        D_LossFunction_D_b = 2 * (y - (a + b * x + c * (x ** 2))) * (-x)
        D_LossFunction_D_c = 2 * (y - (a + b * x + c * (x ** 2))) * -(x ** 2)

        a = a - LearningRate * D_LossFunction_D_a
        b = b - LearningRate * D_LossFunction_D_b
        c = c - LearningRate * D_LossFunction_D_c

    AllLossFunctionPerLearningRate.append(list(AllLossFunction))
    AllLossFunction.clear()

print('\nAllLossFunctionPerLearningRate for Learning rate = 1:\n', AllLossFunctionPerLearningRate[0])
print('\nAllLossFunctionPerLearningRate for Learning rate = 0.1:\n', AllLossFunctionPerLearningRate[1])
print('\nAllLossFunctionPerLearningRate for Learning rate = 0.01:\n', AllLossFunctionPerLearningRate[2])


LossFunctionForIter_1 = AllLossFunctionPerLearningRate[0]
LossFunctionForIter_0point1 = AllLossFunctionPerLearningRate[1]
LossFunctionForIter_0point01 = AllLossFunctionPerLearningRate[2]

xLine = np.linspace(0, iter, iter)
yLine_1 = np.array(LossFunctionForIter_1)
yLine_0point1 = np.array(LossFunctionForIter_0point1)
yLine_0point01 = LossFunctionForIter_0point01

plt.figure()
# plt.plot(xLine, yLine_1)
# plt.plot(xLine, yLine_0point1)
plt.plot(xLine, yLine_0point01)
plt.show()
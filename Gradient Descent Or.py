import numpy as np
import matplotlib.pyplot as plt

LearningRates = [1, 0.1, 0.01]
x = np.array([0, 1 ,2])
y = np.array([1, 3, 7])
iter = 200
Momentum = 0.5   ##   0 < Momentum < 1

AllLossFunction = []
AllLossFunctionPerLearningRate = []

AllLossFunctionMomentum = []
AllLossFunctionPerLearningRateMomentum = []


for LearningRate in LearningRates:
    a, b, c = 2, 2, 0
    aMomentum, bMomentum, cMomentum = a, b, c
    u_a, u_b, u_c = 0, 0, 0
    for _ in range(iter):
        LossFunction = np.float64( (1 / len(x)) * sum((y - (a + b * x + c * (x ** 2))) ** 2) )
        AllLossFunction.append(LossFunction)

        D_LossFunction_D_a = 2 * (y - (a + b * x + c * (x ** 2))) * (-1)
        D_LossFunction_D_b = 2 * (y - (a + b * x + c * (x ** 2))) * (-x)
        D_LossFunction_D_c = 2 * (y - (a + b * x + c * (x ** 2))) * -(x ** 2)

        a = a - LearningRate * D_LossFunction_D_a
        b = b - LearningRate * D_LossFunction_D_b
        c = c - LearningRate * D_LossFunction_D_c

        ##-------------Momentum--------------------------------

        LossFunctionMomentum = np.float64((1 / len(x)) * sum((y - (aMomentum + bMomentum * x + cMomentum * (x ** 2))) ** 2))
        AllLossFunctionMomentum.append(LossFunctionMomentum)

        D_LossFunction_D_a_Momentum = 2 * (y - (aMomentum + bMomentum * x + cMomentum * (x ** 2))) * (-1)
        D_LossFunction_D_b_Momentum = 2 * (y - (aMomentum + bMomentum * x + cMomentum * (x ** 2))) * (-x)
        D_LossFunction_D_c_Momentum = 2 * (y - (aMomentum + bMomentum * x + cMomentum * (x ** 2))) * -(x ** 2)

        u_a = Momentum * u_a + LearningRate * D_LossFunction_D_a_Momentum
        u_b = Momentum * u_b + LearningRate * D_LossFunction_D_b_Momentum
        u_c = Momentum * u_c + LearningRate * D_LossFunction_D_c_Momentum

        aMomentum = aMomentum - u_a
        bMomentum = bMomentum - u_b
        cMomentum = cMomentum - u_c

    AllLossFunctionPerLearningRate.append(list(AllLossFunction))
    AllLossFunction.clear()

    AllLossFunctionPerLearningRateMomentum.append(list(AllLossFunctionMomentum))
    AllLossFunctionMomentum.clear()

print('\nAllLossFunctionPerLearningRate for Learning rate = 1:\n', AllLossFunctionPerLearningRate[0])
print('\nAllLossFunctionPerLearningRate for Learning rate = 0.1:\n', AllLossFunctionPerLearningRate[1])
print('\nAllLossFunctionPerLearningRate for Learning rate = 0.01:\n', AllLossFunctionPerLearningRate[2])

print('\nAllLossFunctionPerLearningRateMomentum for Learning rate = 1:\n', AllLossFunctionPerLearningRateMomentum[0])
print('\nAllLossFunctionPerLearningRateMomentum for Learning rate = 0.1:\n', AllLossFunctionPerLearningRateMomentum[1])
print('\nAllLossFunctionPerLearningRateMomentum for Learning rate = 0.01:\n', AllLossFunctionPerLearningRateMomentum[2])


LossFunctionForIter_1 = AllLossFunctionPerLearningRate[0]
LossFunctionForIter_0point1 = AllLossFunctionPerLearningRate[1]
LossFunctionForIter_0point01 = AllLossFunctionPerLearningRate[2]

LossFunctionForIterMomentum_1 = AllLossFunctionPerLearningRateMomentum[0]
LossFunctionForIterMomentum_0point1 = AllLossFunctionPerLearningRateMomentum[1]
LossFunctionForIterMomentum_0point01 = AllLossFunctionPerLearningRateMomentum[2]

xLine = np.linspace(0, iter, iter)

yLine_1 = np.array(LossFunctionForIter_1)
yLine_0point1 = np.array(LossFunctionForIter_0point1)
yLine_0point01 = np.array(LossFunctionForIter_0point01)

yLineMomentum_1 = np.array(LossFunctionForIterMomentum_1)
yLineMomentum_0point1 = np.array(LossFunctionForIterMomentum_0point1)
yLineMomentum_0point01 = np.array(LossFunctionForIterMomentum_0point01)

plt.figure()
# plt.plot(xLine, yLine_1)
# plt.plot(xLine, yLine_0point1)
plt.plot(xLine, yLine_0point01, 'r')

# plt.plot(xLine, yLine_1)
# plt.plot(xLine, yLine_0point1)
plt.plot(xLine, yLineMomentum_0point01, 'b')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import colorama
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style
colorama.init(autoreset = True)


LearningRates = [1, 0.1, 0.01]
x = np.array([0, 1 ,2])
y = np.array([1, 3, 7])
iter = 400
Momentum = 0.5   ##   0 < Momentum < 1

AllLossFunction = []
AllLossFunctionPerLearningRate = []

AllLossFunctionMomentum = []
AllLossFunctionPerLearningRateMomentum = []

SaveTheta = []
SaveThetaMomentum = []

for LearningRate in LearningRates:
    a, b, c = 2, 2, 0
    aMomentum, bMomentum, cMomentum = a, b, c
    u_a, u_b, u_c = 0, 0, 0
    SaveThetaLocal = []
    SaveThetaMomentumLocal = []
    for _ in range(iter):
        LossFunction = np.float64((1 / (2 * len(x))) * sum(((a + b * x + c * (x ** 2)) - y) ** 2))
        AllLossFunction.append(LossFunction)

        D_LossFunction_D_a = (1 / len(x)) * sum(((a + b * x + c * (x ** 2)) - y) * (1))
        D_LossFunction_D_b = (1 / len(x)) * sum(((a + b * x + c * (x ** 2)) - y) * (x))
        D_LossFunction_D_c = (1 / len(x)) * sum(((a + b * x + c * (x ** 2)) - y) * (x ** 2))

        a = a - LearningRate * D_LossFunction_D_a
        b = b - LearningRate * D_LossFunction_D_b
        c = c - LearningRate * D_LossFunction_D_c

        SaveThetaLocal = [a, b, c]
        ##-------------Momentum--------------------------------

        LossFunctionMomentum = np.float64((1 / (2 *len(x))) * sum((y - (aMomentum + bMomentum * x + cMomentum * (x ** 2))) ** 2))
        AllLossFunctionMomentum.append(LossFunctionMomentum)

        D_LossFunction_D_a_Momentum = (1 / len(x)) * sum(((aMomentum + bMomentum * x + cMomentum * (x ** 2)) - y) * (1))
        D_LossFunction_D_b_Momentum = (1 / len(x)) * sum(((aMomentum + bMomentum * x + cMomentum * (x ** 2)) - y) * (x))
        D_LossFunction_D_c_Momentum = (1 / len(x)) * sum(((aMomentum + bMomentum * x + cMomentum * (x ** 2)) - y) * (x ** 2))

        u_a = Momentum * u_a + LearningRate * D_LossFunction_D_a_Momentum
        u_b = Momentum * u_b + LearningRate * D_LossFunction_D_b_Momentum
        u_c = Momentum * u_c + LearningRate * D_LossFunction_D_c_Momentum

        aMomentum = aMomentum - u_a
        bMomentum = bMomentum - u_b
        cMomentum = cMomentum - u_c

        SaveThetaMomentumLocal = [aMomentum, bMomentum, cMomentum]

    AllLossFunctionPerLearningRate.append(list(AllLossFunction))
    AllLossFunction.clear()

    AllLossFunctionPerLearningRateMomentum.append(list(AllLossFunctionMomentum))
    AllLossFunctionMomentum.clear()

    SaveTheta.append(SaveThetaLocal)
    SaveThetaMomentum.append(SaveThetaMomentumLocal)


# print('\nAllLossFunctionPerLearningRate for Learning rate = 1:\n', AllLossFunctionPerLearningRate[0])
# print('\nAllLossFunctionPerLearningRateMomentum for Learning rate = 1:\n', AllLossFunctionPerLearningRateMomentum[0])
#
# print('\nAllLossFunctionPerLearningRate for Learning rate = 0.1:\n', AllLossFunctionPerLearningRate[1])
# print('\nAllLossFunctionPerLearningRateMomentum for Learning rate = 0.1:\n', AllLossFunctionPerLearningRateMomentum[1])
#
# print('\nAllLossFunctionPerLearningRateMomentum for Learning rate = 0.01:\n', AllLossFunctionPerLearningRateMomentum[2])
# print('\nAllLossFunctionPerLearningRate for Learning rate = 0.01:\n', AllLossFunctionPerLearningRate[2])


print(Fore.RED + '----------------------------------Loss Function-------------------------------------------------')


print('\n The a,b,c - Theta0, Theta1, Theta2 in the Loss Function after {} iter: (1 / (2 * len(x))) * sum(((a + b * x + c * (x ** 2)) - y) ** 2) for Learning Rate {} are:{} {} {}'
.format(iter, LearningRates[1], SaveTheta[1][0], SaveTheta[1][1], SaveTheta[1][2]))
print('\n The a,b,c - Theta0, Theta1, Theta2 in the Loss Function after {} iter: (1 / (2 * len(x))) * sum(((a + b * x + c * (x ** 2)) - y) ** 2) for Learning Rate {} are:{} {} {}\n'
.format(iter, LearningRates[2], SaveTheta[2][0], SaveTheta[2][1], SaveTheta[2][2]))

print(Fore.RED + '----------------------------------Loss Function With Momentum-------------------------------------------------')

print('\n The a,b,c - Theta0, Theta1, Theta2 in the Loss Function With Momentum {} after {} iter: (1 / (2 * len(x))) * sum(((a + b * x + c * (x ** 2)) - y) ** 2) for Learning Rate {} are:{} {} {}'
.format(Momentum, iter, LearningRates[1], SaveThetaMomentum[1][0], SaveThetaMomentum[1][1], SaveThetaMomentum[1][2]))
print('\n The a,b,c - Theta0, Theta1, Theta2 in the Loss Function With Momentum {} after {} iter: (1 / (2 * len(x))) * sum(((a + b * x + c * (x ** 2)) - y) ** 2) for Learning Rate {} are:{} {} {}\n'
.format(Momentum, iter, LearningRates[2], SaveThetaMomentum[2][0], SaveThetaMomentum[2][1], SaveThetaMomentum[2][2]))





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
plt.plot(xLine, yLine_0point1, 'r')
plt.plot(xLine, yLine_0point01, 'g')

# plt.plot(xLine, yLineMomentum_1)
plt.plot(xLine, yLineMomentum_0point1, 'r--')
plt.plot(xLine, yLineMomentum_0point01, 'g--')

plt.xlabel('Iter')
plt.ylabel('Loss Function')
plt.title('Loss Function Per Iter')

plt.legend(['LearningRates = 0.1', 'LearningRates = 0.01', 'LearningRates = 0.1 with Momentum = 0.5', 'LearningRates = 0.01 with Momentum = 0.5'])
plt.show()
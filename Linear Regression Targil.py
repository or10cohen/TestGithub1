import numpy as np
import random
import colorama
from colorama import Fore, Style
colorama.init(autoreset = True)


Integer10NumbersRand = list(np.random.randint(0, 100, 10))
print(Fore.RED + '\nsection a. 10 random integer numbers: \n', Integer10NumbersRand)
x = np.random.uniform(0, 20, 10)
print(Fore.RED + '\nsection b. 10 random float numbers "x": \n', x, type(x))

Vector5Numbers3 = []
for i in range(0, 5):
    n = random.randint(1, 10) * 3
    Vector5Numbers3.append(n)

print(Fore.RED + '\nsection c. vecotr 5 random numbers multiply by 3: \n', Vector5Numbers3)

VectorFibonacci = [1, 1]
for i in range(0, 10):
    l = VectorFibonacci[i] + VectorFibonacci[i + 1]
    VectorFibonacci.append(l)

print(Fore.RED + '\nsection c. random number from Fibonacci: \n', VectorFibonacci)
print(Fore.RED + '\nthe random number: ', random.choice(VectorFibonacci))

##--------------------------------------------------------------------------------------------------------------

Constant1 = random.choice(VectorFibonacci)
y_line_1 = Constant1 * x                         ###### oren
print(Fore.RED + f'\nsection f. create y_line_1 = {Constant1}(Constant) * x[i]:\n', y_line_1)

GaussianNoise1 = np.random.normal(size=(10))
print(Fore.RED + '\nsection g1. GaussianNoise1:\n', GaussianNoise1)

y_line_1PlusGaussian = y_line_1 + GaussianNoise1
print(Fore.RED + f'\nsection g1. create y_line_1PlusGaussian = {Constant1}(Constant) * x[i] + GaussianNoise1[i]:\n', y_line_1PlusGaussian)

##--------------------------------------------------------------------------------------------------------------
Constant2 = random.choice(VectorFibonacci)
Constant3 = random.choice(VectorFibonacci)
y_line_2 = Constant2 * x + Constant3
print(Fore.RED + f'\nsection h1. create y_line_2 = {Constant2}(Constant) * x[i] + {Constant3}(Constant):\n', y_line_2)

GaussianNoise2 = np.random.normal(size=(10))
print(Fore.RED + '\nsection h2. GaussianNoise2:\n', list(GaussianNoise2))

y_line_2PlusGaussian = y_line_2 + GaussianNoise2
print(Fore.RED + f'\nsection h3. create y_line_2PlusGaussian = {Constant2}(Constant) * x[i] + {Constant3}(Constant) + GaussianNoise2[i]:\n', y_line_2PlusGaussian)

##--------------------------------------------------------------------------------------------------------------
Constant4 = random.choice(VectorFibonacci)
Constant5 = random.choice(VectorFibonacci)
Constant6 = random.choice(VectorFibonacci)
y_line_3 = Constant4 * x ** 2 + Constant5 * x + Constant6
print(Fore.RED + f'\nsection i1. create y_line_3 = {Constant4}(Constant) * x[i] ** 2 + {Constant5}(Constant) * x[i] + {Constant6}(Constant):\n', y_line_3)

GaussianNoise3 = np.random.normal(size=(10))
print(Fore.RED + '\nsection i2. GaussianNoise3:\n', list(GaussianNoise3))

y_line_3PlusGaussian = y_line_3 + GaussianNoise3
print(Fore.RED + f'\nsection i3. create y_line_3PlusGaussian = {Constant4}(Constant) * x[i] ** 2 + {Constant5}(Constant) * x[i] + {Constant6}(Constant) + GaussianNoise3[i]:\n', y_line_3PlusGaussian)

##------------------------------------------------------------------------------------------------------------------------------
print(Fore.BLUE + '\nin sections g, h, i we create 3 DataSet that have the same "x" and y are description in 3 equations.')

print(Fore.RED + '\nThe vector "x" is:\n', x)
print(Fore.RED + f'\n1. y_line_1PlusGaussian = {Constant1}(Constant) * x[i] + GaussianNoise1[i]\n' , y_line_1PlusGaussian)
print(Fore.RED + f'2. y_line_2PlusGaussian = {Constant2}(Constant) * x[i] + {Constant3}(Constant) + GaussianNoise2[i]\n', y_line_2PlusGaussian)
print(Fore.RED + f'3. y_line_3PlusGaussian = {Constant4}(Constant) * x[i] ** 2 + {Constant5}(Constant) * x[i] + {Constant6}(Constant) + GaussianNoise3[i]\n', y_line_3PlusGaussian)


#######----------------------------------Normal Equation--------------------------------------------------------------------------
#######-----------------y_line_1PlusGaussian = 4(Constant) * x[i] + GaussianNoise1[i]---------------------------------------------
print(Fore.BLUE + f'\n\n\n how too solve the first equation: ***** y_line_1PlusGaussian = {Constant1}(Constant) * x[i] + GaussianNoise1[i]  *****\n')
x = x.reshape(len(x), 1)
print(Fore.RED + 'vector x(10)\n', x)
print(Fore.RED + '\nvector y(10, y_line_1PlusGaussian)\n' , y_line_1PlusGaussian)
y_line_1 = y_line_1PlusGaussian
h = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y_line_1) # normal equation
# h[0] = Theta1
print(Fore.RED + f'\nThe result of Normal equation for y_line_1PlusGaussian = {Constant1}(Constant) * x[i] + GaussianNoise1[i] \nare describe by the formula: h = (x^t * x)^(-1) * x^t * y\n', h)

#######-----------------y_line_2PlusGaussian = 4(Constant) * x[i] + 4(Constant) + GaussianNoise2[i]---------------------------------------------
print(Fore.BLUE + f'\n\n\nhow too solve the second equation: ***** y_line_2PlusGaussian = {Constant2}(Constant) * x[i] + {Constant3}(Constant) + GaussianNoise2[i]  *****\n')

b = np.ones(len(x))
print(Fore.RED + 'create ones vector length of vector x(10)\n', b)
print(Fore.RED + '\nvector y(10, y_line_2PlusGaussian)\n' , y_line_2PlusGaussian)
print(Fore.RED + '\nvector x(10)\n', x)
X = np.column_stack((x, b))
print(Fore.RED + '\ncreate Matrix from vector x and vector b. X = np.column_stack((x, b))\n', X)
# print(Fore.RED + '\nx shape\n', x.shape)
# print(Fore.RED + '\ny shape\n', np.array(y_line_2PlusGaussian).shape)
# print(Fore.RED + '\nX shape\n', X.shape)
# print(Fore.RED + '\nX^t shape\n', np.transpose(X).shape)
y_line_2 = y_line_2PlusGaussian
h = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y_line_2) # normal equation
# h[0] = Theta1
# h[1] = Theta2
print(Fore.RED + f'\nThe result of Normal equation for y_line_2PlusGaussian = {Constant2}(Constant) * x[i] + {Constant3}(Constant) + GaussianNoise2[i] are describe by the formula: h = (x^t * x)^(-1) * x^t * y\n', h)

#######-----------------y_line_3PlusGaussian = 4(Constant) * x[i] ** 2 + 4(Constant) * x[i] + 4(Constant) + GaussianNoise3[i]---------------------------------------------
print(Fore.BLUE + f'\n\n\nhow too solve the thired equation: ***** y_line_3PlusGaussian = {Constant4}(Constant) * x[i] ** 2 + {Constant5}(Constant) * x[i] + {Constant6}(Constant) + GaussianNoise3[i]  *****\n')

x2 = x * x
print(Fore.RED + 'create x^2 vector from vector x(10)\n', x2)
b = np.ones(len(x))
print(Fore.RED + 'create ones vector length of vector x(10)\n', b)
print(Fore.RED + '\nvector y(10, y_line_2PlusGaussian)\n' , y_line_3PlusGaussian)
print(Fore.RED + '\nvector x(10)\n', x)
X = np.column_stack((x2, x, b))
print(Fore.RED + '\ncreate Matrix from vector x^2, x and vector b. X = np.column_stack((x^2, x, b))\n', X)
# print(Fore.RED + '\nx shape\n', x.shape)
# print(Fore.RED + '\ny shape\n', np.array(y_line_2PlusGaussian).shape)
# print(Fore.RED + '\nX shape\n', X.shape)
# print(Fore.RED + '\nX^t shape\n', np.transpose(X).shape)
y_line_3 = y_line_3PlusGaussian
h = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y_line_3) # normal equation
# h[0] = Theta1
# h[1] = Theta2
# h[2] = Theta3
print(Fore.RED + f'\nThe result of Normal equation for y_line_3PlusGaussian = {Constant4}(Constant) * x[i] ** 2 + {Constant5}(Constant) * x[i] + {Constant6}(Constant) + GaussianNoise3[i] are describe by the formula: h = (x^t * x)^(-1) * x^t * y\n', h)




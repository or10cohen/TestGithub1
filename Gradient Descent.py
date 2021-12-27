import numpy as np
import matplotlib.pyplot as plt
from Tools.scripts.nm2def import symbols
# sum_loss_function_matrix = np.zeros((3, 200))
sum_loss_function = []
total_sum_loss_function = np.empty((0, 200), float)
print(total_sum_loss_function.shape)
lr = [0.01, 0.1, 1]
for i in lr:
    print(i)
    g = 0
    a = 2
    b = 2
    c = 0
    x = [0, 1, 2]
    x = np.array(x)
    y = [1, 3, 7]
    y = np.array(y)
    # hypothesis class h(x) = a + b*x + c*x**2
    f = 200
    while f > 0:
        loss_function = sum((y - (a + b * x + c * (x ** 2))) ** 2)
        sum_loss_function.append(loss_function)

        dl_da = -2 * (y - (a + b * x + c * (x ** 2)))
        dl_db = 2 * (y - (a + b * x + c * (x ** 2))) * (-x)
        dl_dc = 2 * (y - (a + b * x + c * (x ** 2))) * (-2 * x * c)

        step_size_a = i * dl_da
        step_size_b = i * dl_db
        step_size_c = i * dl_dc

        a = a - step_size_a
        b = b - step_size_b
        c = c - step_size_c
        f = f - 1
    print(sum_loss_function)
    print(len(sum_loss_function))
    h = np.array(sum_loss_function).reshape((1, -1))
    new_array = np.array(sum_loss_function)
    total_sum_loss_function = np.concatenate((total_sum_loss_function, h), axis=0)
    sum_loss_function.clear()

print(total_sum_loss_function.shape)
plt.figure()
plt.plot(np.linspace(0, 100, 100), total_sum_loss_function[0, :100])
plt.plot(np.linspace(0, 100, 100), total_sum_loss_function[1, :100], 'r')
plt.legend(['lr=0.01', 'lr=0.1'])
plt.show()

plt.figure()
plt.plot(np.linspace(0, 100, 100), total_sum_loss_function[2, :100])
plt.legend(['lr=1'])
plt.show()

# Learning rate 0.1 and momentum 0.9

sum_loss_function_new =[]
a = 2
b = 2
c = 0
v1 = 0
v2 = 0
v3 = 0
x = [0, 1, 2]
x = np.array(x)
y = [1, 3, 7]
y = np.array(y)

f = 200
while f > 0:
    loss_function = sum((y - (a + b * x + c * (x * 2))) * 2)
    sum_loss_function_new.append(loss_function)

    dl_da = -2 * (y - (a + b * x + c * (x ** 2)))
    dl_db = 2 * (y - (a + b * x + c * (x ** 2))) * (-x)
    dl_dc = 2 * (y - (a + b * x + c * (x ** 2))) * (-2 * x * c)

    step_size_a = 0.1 * dl_da
    step_size_b = 0.1 * dl_db
    step_size_c = 0.1 * dl_dc

    v1_new = 0.9 * v1 + step_size_a
    v2_new = 0.9 * v2 + step_size_b
    v3_new = 0.9 * v3 + step_size_c

    a = a - v1_new
    b = b - v2_new
    c = c - v3_new
    f = f - 1

    v1 = v1_new
    v2 = v2_new
    v3 = v3_new

plt.figure()
plt.plot(np.linspace(0, 100, 100), total_sum_loss_function[1, :100])
plt.plot(np.linspace(0, 100, 100), sum_loss_function_new[:100], 'r')
plt.legend(['No momentum', 'with momentum'])
plt.show()
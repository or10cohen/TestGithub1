# https://www.tutorialspoint.com/sympy/sympy_derivative.htm

import numpy as np
import sympy as sym


LR = 0.1
i = 1.5
w_t0 =  0.8

w = sym.Symbol('w')  #x, y, z = sym.Symbol('x y z')
a_0 = w * i
CF = (a_0 - 0.5)**2

dCF_dw = CF.diff(w)  # dCF_dw = CF.diff(CF, w)
# print(dCF_dw)
dCF_dw_define_w = dCF_dw.subs(w, w_t0)
# print(dCF_dw_define_w)

all_w = [w_t0]
all_CF = [(w_t0 * i - 0.5)**2]
Flag = True
while (all_CF[-1] > 0.00001) and (dCF_dw.subs(w, w_t0) > 0):
    w_t1 = w_t0 - 0.1 * dCF_dw.subs(w, w_t0)
    w_t0 = w_t1
    all_w.append(w_t1)
    all_CF.append((w_t0 * i - 0.5)**2)

print('weight per iteration', all_w)
print('Cost Function per w', all_CF)




# w_t1 = w + LR*



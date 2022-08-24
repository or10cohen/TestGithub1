# https://www.tutorialspoint.com/sympy/sympy_derivative.htm

import numpy as np
from sympy import *


LR = 0.1
i = 1.5
w_t0, w_t1, w_t2 = 0.5, 0.5, 0.5

w0, w1, w2 = Symbol('w0 w1 w2')  #x, y, z = sym.Symbol('x y z')
a_0 = w0 * w1 * w2 * i
CF = (a_0 - 0.5)**2

dCF_dw0 = CF.diff(CF, w0)  # dCF_dw = CF.diff(CF, w)
dCF_dw1 = CF.diff(CF, w1)
dCF_dw2 = CF.diff(CF, w2)
print(dCF_dw0)
print(dCF_dw1)
print(dCF_dw2)

dCF_dw0_define_w0 = dCF_dw0.subs(w0, w_t0)
dCF_dw1_define_w1 = dCF_dw1.subs(w1, w_t1)
dCF_dw2_define_w2 = dCF_dw2.subs(w2, w_t2)
print(dCF_dw0_define_w0)
print(dCF_dw1_define_w1)
print(dCF_dw2_define_w2)

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



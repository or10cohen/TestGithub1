# https://www.tutorialspoint.com/sympy/sympy_derivative.htm
import numpy as np
from sympy import *

LR = 0.1
inp = 1.5
w0_t0, w1_t0, w2_t0 = 0.5, 0.5, 0.5

w0, w1, w2 = symbols('w0 w1 w2')  #x, y, z = sym.Symbol('x y z')
a_0 = w0 * w1 * w2 * inp
CF = (1 / 3) * ((w0 * inp - 0.5)**2 + (w1 * inp - 0.5)**2 + (w2 * inp - 0.5)**2)

dCF_dw0 = diff(CF, w0)  # dCF_dw = CF.diff(CF, w)
dCF_dw1 = diff(CF, w1)
dCF_dw2 = diff(CF, w2)
dCF_dw = [dCF_dw0, dCF_dw1, dCF_dw2]
print(dCF_dw0)
print(dCF_dw1)
print(dCF_dw2)

dCF_dw0_define_w0 = dCF_dw0.subs(w0, w0_t0)
dCF_dw1_define_w1 = dCF_dw1.subs(w1, w0_t0)
dCF_dw2_define_w2 = dCF_dw2.subs(w2, w0_t0)
dCF_dw_define_w = [dCF_dw0_define_w0, dCF_dw1_define_w1, dCF_dw2_define_w2]
print(dCF_dw0_define_w0)
print(dCF_dw1_define_w1)
print(dCF_dw2_define_w2)


w = [w0, w1, w2]
all_w = [[w0_t0], [w1_t0], [w2_t0]]
all_CF = [(1 / 3) * ((w0_t0 * inp - 0.5)**2 + (w0_t0 * inp - 0.5)**2 + (w0_t0 * inp - 0.5)**2)]
print(all_w)
print(all_CF)

while all_CF[-1] > 0.001: # and ((dCF_dw1_define_w1 > 0) or (dCF_dw1_define_w1) or (dCF_dw2_define_w2)):
    i = 0
    for i in range(3):
        print(i)
        w_tnext = all_w[i][-1] - 0.1 * dCF_dw[i].subs(w[i], all_w[i][-1])
        all_w[i].append(w_tnext)
        all_w[i][-1] = w_tnext
        i += 1
    print(all_w)
    all_CF.append((1 / 3) * ((all_w[0][-1] * inp - 0.5) ** 2 + (all_w[1][-1] * inp - 0.5) ** 2 + (all_w[2][-1] * inp - 0.5) ** 2))
    print(all_CF[-1])

print('weight per iteration', all_w)
print('Cost Function per w', all_CF)


# w_t1 = w + LR*



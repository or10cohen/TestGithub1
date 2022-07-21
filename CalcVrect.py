import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

def CalcVrect(Vin, Lp, Ls, Rp, Rs, wp, ws, w, k, Rl, DutyC):
    MaxH, qr, Div = 8, 0, 50
    ZeroMatrix = np.zeros((100, int(((MaxH * 2 + 1) + 1) / 2)),  dtype=complex)
    I2C = ZeroMatrix
    I1V = ZeroMatrix
    I1C = ZeroMatrix
    I2V = ZeroMatrix
    VrectA = Iswitch = np.zeros(100)
    for iter in range(1): #range(1, 5):
        for q in range(1, 101):
            qa = (qr + q / Div) * round(np.pi, 4)
            qa = round(qa, 4)
            for n in range(1, MaxH * 2 + 2, 2):
                # Matrix A * I1n B * I2n = D
                # Matrix B * I1n C * I2n = E*Vrect
                Yp = 1 - np.square(wp / (n * w))
                Ys = 1 - np.square(ws / (n * w))
                A = Rp + (w * n * Lp * Yp) * 1j
                B = (w * n * np.sqrt(Lp * Ls) * k) * 1j
                C = Rs + (w * n * Ls * Ys) * 1j
                D = 4 * np.sin(round(np.pi, 4) * DutyC * n) * Vin * np.exp((qa * n) * 1j) / (round(np.pi, 4) * n)
                E = 4 * np.sin(np.pi / 2 * n) / (np.pi * n)  ################## 2 * n
                #############----------------------debug until here !!!except qa-D!!-----------------------
                print(q - 1, int((n + 1) / 2 - 1))
                print((E * A) / (A * C - np.square(B)), I2V[q - 1, int((n + 1) / 2 - 1)])
                # I2n = EA / (AC - B ^ 2) * Vrect - DB / (AC - B ^ 2) = I2V(n) * Vrect + I2C(n)
                I2V[q - 1, int((n + 1) / 2 - 1)] = (E * A) / (A * C - np.square(B))
                I2C[q - 1, int((n + 1) / 2 - 1)] = -D * B / (A * C - np.square(B))
                I1V[q - 1, int((n + 1) / 2 - 1)] = -E * B / (A * C - np.square(B))
                I1C[q - 1, int((n + 1) / 2 - 1)] = D * C / (A * C - np.square(B))
                if n == 1 - 1:
                    C = Rs + (w * n * Ls * Ys + Rl * 8) * 1j / np.square(np.pi)
                    I2E = -D * B / (A * C - np.square(B))
                    Vrect2 = I2E * Rl * 2 / np.pi
                    #Vrect2=I2E*Rl*8/(pi()^2*sqrt(2));


            #Sum(Real(2 * I2n / (pi * n))) = Vrect / Rl
            #Vrect = sum(real(2 * I2C(n) / (pi() * n))) / (1 / Rl - sum(real(2 * I2V(n) / (pi() * n))))
            n = np.arange(1, MaxH * 2 + 2, 2)
            # print(q, n)
            # print(np.sum(np.real(2 * I2C[q - 1, :] * np.sin(np.pi / 2 * n) / (np.pi * n))), np.sum(np.real(2 * I2V[q - 1, :] * np.sin(np.pi / 2 * n) / (np.pi * n))))
            VrectA[q - 1] = - np.sum(np.real(2 * I2C[q - 1, :] * np.sin(np.pi / 2 * n) / (np.pi * n))) \
                    / (1 / Rl + np.sum(np.real(2 * I2V[q - 1, :] * np.sin(np.pi / 2 * n) / (np.pi * n))))
            # print(q, n)
            # print(np.sum(np.imag(I2C[q - 1, :] * np.sin(np.pi / 2 * n))), np.sum(np.imag(I2V[q - 1, :] * np.sin(np.pi / 2 * n))), VrectA[q - 1])
            Iswitch[q - 1] = np.sum(np.imag(I2C[q - 1, :] * np.sin(np.pi / 2 * n))) + np.sum(np.imag(I2V[q - 1, :] \
                    * np.sin(np.pi / 2 * n))) * VrectA[q - 1]
            if VrectA[q - 1] < 0:
                Iswitch[q - 1] = 100

        v = min(np.absolute(Iswitch))
        l = np.where(Iswitch == v)
        Vrect = VrectA[l]
        # print(Iswitch, '\n', v, '\n', l)
        # print(I1V.shape, '\n', VrectA, '\n', l)
        I1 = I1C[l, :] + I1V[l, :] * Vrect
        I2 = I2C[l, :] + I2V[l, :] * Vrect
        n = np.arange(1, MaxH * 2 + 2, 2)
        Pout = np.sum(np.real(I2 * np.sin(np.pi / 2 * n)) * Vrect * 2 / (n * np.pi))
        qa = ((qr + l / Div) * np.pi)
        Pin = -sum(np.real(I1 * 2. * np.sin(np.pi * DutyC * n) * Vin * np.exp(-i * qa * n) / (np.pi * n)))
        Prp = sum(abs(I1) ^ 2) * Rp / 2
        Prs = sum(abs(I2) ^ 2) * Rs / 2
        Eff = Pout / Pin
        qr = qr + (l - 0.5) / Div
        Div = Div * 100
    return Vrect, I1, I2, Vrect2, Eff, Pin, Pout, Prp, Prs






if __name__ == '__main__':
    Vin = 48
    Lp = 18.86 * (10 ** -6)
    Ls = 18.9 * (10 ** -6)
    Rp = 0.1
    Rs = 0.1
    Cp = 139 * (10 ** -9)
    Cs = 103 * (10 ** -9)
    wp = 1 / np.sqrt(Cp * Lp)
    ws = 1 / np.sqrt(Cs * Ls)
    w = (101 * (10 ** 3)) * 2 * round(np.pi, 4)
    k = 0.175
    Rminload = 570
    DutyC = 0.41

    FresP = wp * 1 / (2 * np.pi)
    FresS = ws / (2 * np.pi)
    Vout = 25.5
    Pout = 60
    Rload = (Vout ** 2) / Pout

    #CalcVrect(Vin, Lp, Ls, Rp, Rs, wp, ws, w, k, Rminload, DutyC)

    Or = CalcVrect(Vin, Lp, Ls, Rp, Rs, wp, ws, w, k, Rminload, DutyC)

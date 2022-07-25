import numpy as np
import matplotlib.pyplot as plt
import CalcVrect

# Varona project
# this script simulate different coupling (range from 0.175-0.49).
# High coupling for Z-gap of 10[mm] and XY misalignment of 12[mm].
# Low coupling for Z-gap of 38[mm] and no XY misalignment.

def Varona(Vin, Lp, Ls, Rp, Rs, wp, ws, k, Rload, RminLoad, DutyC):
    VrectA, VrectB, I1A, I2A, I1B, I2B, I1A1, I2A1, EffA = np.zeros(101), np.zeros(101), np.zeros(101), np.zeros(101)\
        , np.zeros(101), np.zeros(101), np.zeros(101), np.zeros(101), np.zeros(101)
    f = np.arange(0, 101) # f = 0:100
    for n in range(len(f)):
        w = (f[0] + (n + 1) - 1 + 100) * (10 ** 3) * 2 * np.pi
        Vrect, I1, I2, Vrect2, Eff, Pin, Pout, Prp, Prs = CalcVrect.CalcVrect(Vin, Lp, Ls, Rp, Rs, wp, ws, w, k, Rload, DutyC)
        VrectA[n] = np.abs(Vrect)
        I1A[n] = np.sqrt(sum(np.abs(I1) ** 2) / 2)
        I2A[n] = np.sqrt(sum(np.abs(I2) ** 2) / 2)
        I1A1[n] = np.abs(I1[2])
        I2A1[n] = np.abs(I2[2])
        EffA[n] = (np.abs(Pout)) / (np.abs(Pin) + Pidle)

        # minimum load
        Vrect, I1, I2, Vrect2, Eff, Pin, Pout, Prp, Prs = CalcVrect.CalcVrect(Vin, Lp, Ls, Rp, Rs, wp, ws, w, k, RminLoad, DutyC)
        VrectB[n] = np.abs(Vrect)
        I1B[n] = np.sqrt(sum(np.abs(I1) ** 2) / 2)
        I2B[n] = np.sqrt(sum(np.abs(I2) ** 2) / 2)
    return f, VrectA, VrectB, EffA, I1A, I2A, I1A1, I2A1, I1B, I2B

def plot(f, VrectA, VrectB, EffA, I1A, I2A, I1A1, I2A1, I1B, I2B):
    fig1, ax1 = plt.subplots()
    fig2, ax2= plt.subplots()
    fig3, ax3= plt.subplots()
    fig4, ax4 = plt.subplots()

    # fig.figsize(15, 15)
    ax1.set_xlim(100, 200)
    ax1.plot(100 + f, VrectA, 'b', 100 + f, VrectB, 'r', lw=1)
    ax1.set_title('Vrect loaded (200W) vs. Min load - FB DC=20%')
    ax2.set_xlim(100, 200)
    ax2.plot(100 + f, EffA, 'b', lw=1)
    ax2.set_title('Efficency')
    ax3.set_xlim(100, 200)
    ax3.plot(100 + f, I1A, 'b', 100 + f, I2A, 'r', 100 + f, I1A1, 'm', 100 + f, I2A1, 'y', lw=1)
    ax3.set_title('Tx and Rx RMS current Load')
    ax4.set_xlim(100, 200)
    ax4.plot(100 + f, I1B / np.sqrt(2), 'b', 100 + f, I2B / np.sqrt(2), 'r', lw=1)
    ax4.set_title('Tx and Rx RMS current Min Load')
    fig1.savefig('Varona - Vrect loaded (200W) vs. Min load - FB DC = 20 present.png')
    fig2.savefig('Varona - Efficency.png')
    fig3.savefig('Varona - Tx and Rx RMS current Load.png')
    fig4.savefig('Varona - Tx and Rx RMS current Min Load.png')



if __name__ == '__main__':
    Lp = 18.86 * (10 ** -6)  # 21.59 (@k=0.49) %20.55 (@k=0.45) %19.69 (@k=0.35) %19.29 (@k=0.277) %19.07 (@k=0.22) %18.86 (@k=0.175)
    Cp = 139 * (10 ** -9)
    wp = 1 / np.sqrt(Cp * Lp)
    FresP = wp / (2 * np.pi)
    Rp = 0.1
    Vin = 48  # Full-Bridge
    DutyC = 0.4

    Ls = 18.9 * (10 ** -6)  # %21.63 (@k=0.49) %20.64 (@k=0.45) %19.79 (@k=0.35) %19.39 (@k=0.277) %19.14 (@k=0.22) %18.9 (@k=0.175)
    Cs = 103 * (10 ** -9)
    ws = 1 / np.sqrt(Cs * Ls)
    FresS = ws / (2 * np.pi)
    Rs = 0.1

    Vout = 25.5
    Pout = 60
    Rload = (Vout ** 2) / Pout
    RminLoad = 570

    k = 0.175  # %0.49 %0.45 %0.35 %0.277 %0.22 %0.175

    # Rrbt=0.01
    # Rbat=0.02
    Pidle = 0.25
    f, VrectA, VrectB, EffA, I1A, I2A, I1A1, I2A1, I1B, I2B = Varona(Vin, Lp, Ls, Rp, Rs, wp, ws, k, Rload, RminLoad, DutyC)
    plot_Varona = plot(f, VrectA, VrectB, EffA, I1A, I2A, I1A1, I2A1, I1B, I2B)
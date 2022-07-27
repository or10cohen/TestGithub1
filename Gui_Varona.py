import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import Varona_k_range
import CalcVrect
from PIL import Image
# from dash import html
# import dash_bootstrap_components as dbc



def main():
    epsilon = 0.000001
    st.title('Varona')
    with st.sidebar:
        Vin = st.number_input('Insert Vin', value=48)
        st.write('Vin:', Vin)

        Lp = st.number_input('Insert Lp:', value=18.86 * (10 ** -6), format="%.10f")
        st.write('Lp:', Lp)
        Cp = st.number_input('Insert Cp:', value=139 * (10 ** -9), format="%.10f")
        st.write('Cp:', Cp)
        Rp = st.number_input('Insert Rp', value=0.1)
        st.write('Rp:', Rp)
        Ls = st.number_input('Insert Ls:', value=18.9 * (10 ** -6), format="%.10f")
        st.write('Ls:', Ls)
        Cs = st.number_input('Insert Cs:', value=103 * (10 ** -9), format="%.10f")
        st.write('Cs:', Cs)
        Rs = st.number_input('Insert Rs', value=0.1 + epsilon)
        st.write('Rs:', Rs)

        w = st.number_input('Insert w:', value=(101 * (10 ** 3)) * 2 * np.pi, format="%.10f")
        st.write('w:', w)

        DutyC = st.number_input('Insert DutyC', value=0.4)
        st.write('DutyC:', DutyC)

        Vout = st.number_input('Insert Vout', value=25.5, format="%.4f")
        st.write('Vout:', Vout)
        Pout = st.number_input('Insert Pout', value=60)
        st.write('Pout: ', Pout)
        k = st.number_input('Insert k', value=0.175 + epsilon, format="%.4f")
        st.write('k: ',k)

        RminLoad = st.number_input('Insert RminLoad', value=570)
        st.write('RminLoad: ', RminLoad)
        Pidle = st.number_input('Insert Pidle', value=0.25)
        st.write('Pidle: ', Pidle)

        wp = 1 / np.sqrt(Cp * Lp)
        st.write('wp = 1 / np.sqrt(Cp * Lp):\n', wp)
        FresP = wp / (2 * np.pi)
        st.write('FresP = wp / (2 * np.pi):\n', FresP)
        ws = 1 / np.sqrt(Cs * Ls)
        st.write('ws = 1 / np.sqrt(Cs * Ls):\n', ws)
        FresS = ws / (2 * np.pi)
        st.write('FresS = ws / (2 * np.pi):\n', FresS)
        Rload = (Vout ** 2) / Pout
        st.write('Rload = (Vout ** 2) / Pout:', Rload)


        with open("CalcVrect.py") as file:
            btn = st.download_button(
                label="Download CalcVrect Python Resources File",
                data=file,
            )
        with open("Varona_k_range.py") as file:
            btn = st.download_button(
                label="Download Varona_k_range Python Resources File",
                data=file,
            )
        with open("Gui_Varona.py") as file:
            btn = st.download_button(
                label="Download Gui_Varona Python Resources File",
                data=file,
            )

    Vrect, I1, I2, Vrect2, Eff, Pin, Pout, Prp, Prs = CalcVrect.CalcVrect(Vin, Lp, Ls, Rp, Rs, wp, ws, w, k, RminLoad, DutyC)
    f, VrectA, VrectB, EffA, I1A, I2A, I1A1, I2A1, I1B, I2B = Varona_k_range.Varona(Vin, Lp, Ls, Rp, Rs, wp, ws, k, Rload, RminLoad, DutyC, Pidle)
    plot_Varona = Varona_k_range.plot(f, VrectA, VrectB, EffA, I1A, I2A, I1A1, I2A1, I1B, I2B)


    Varona0 = Image.open('Varona - Tx and Rx RMS current Load.png')
    st.image(Varona0, caption='Sunrise by the mountains')
    Varona1 = Image.open('Varona - Tx and Rx RMS current Min Load.png')
    Varona2 = Image.open('Varona - Efficency.png')
    # Varona3 = Image.open('Varona - Vrect loaded (200W) vs. Min load - FB DC = 20 present.png')


    # carousel = dbc.Carousel(
    #     items=[
    #         {"key": "1", "src": "Varona - Vrect loaded (200W) vs. Min load - FB DC = 20 present.png"},
    #         {"key": "2", "src": "Varona - Efficency.png"},
    #         {"key": "3", "src": "Varona - Tx and Rx RMS current Min Load.png"},
    #         {"key": "3", "src": "Varona - Tx and Rx RMS current Min Load.png"},
    #     ],
    #     controls=True,
    #     indicators=True,
    # )

main()
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import Varona_k_range
import CalcVrect

def main():
    epsilon = 000000.1
    st.title('Varona')
    col1, col2 = st.columns([5, 5])
    with st.sidebar:
        Vin = st.number_input('Insert Vin', value=48)
        st.write('Vin:', Vin)

        Lp = st.number_input('Insert Lp', value=18.86 * (10 ** -6))
        st.write('Lp:', Lp)
        Cp = st.number_input('Insert Cp', value=139 * (10 ** -9))
        st.write('Cp:', Cp)
        wp = 1 / np.sqrt(Cp * Lp)
        st.write('wp = 1 / np.sqrt(Cp * Lp):', wp)
        FresP = wp / (2 * np.pi)
        st.write('FresP = wp / (2 * np.pi):', FresP)
        Rp = st.number_input('Insert Rp', value=0.1)
        st.write('Rp:', Rp)

        Ls = st.number_input('Insert Ls', value=18.9 * (10 ** -6))
        st.write('Ls:', Ls)
        Cs = st.number_input('Insert Cs', value=103 * (10 ** -9))
        st.write('Cs:', Cs)
        ws = 1 / np.sqrt(Cs * Ls)
        st.write('ws = 1 / np.sqrt(Cs * Ls):', ws)
        FresS = ws / (2 * np.pi)
        st.write('FresS = ws / (2 * np.pi):', FresS)
        Rs = st.number_input('Insert Rs', value=0.1 + epsilon)
        st.write('Rs:', Rs)

        DutyC = st.number_input('Insert DutyC', value=0.4)
        st.write('DutyC:', DutyC)

        Vout = st.number_input('Insert Vout', value=0.175)
        st.write('Vout:', Vout)
        Pout = st.number_input('Insert Pout', value=60)
        st.write('Pout: ', Pout)
        k = st.number_input('Insert k', value=0.175 + epsilon)
        st.write('k: ',k)
        Rload = (Vout ** 2) / Pout
        st.write('Rload = (Vout ** 2) / Pout:', Rload)
        RminLoad = st.number_input('Insert RminLoad', value=570)
        st.write('RminLoad: ', RminLoad)
        Pidle = st.number_input('Insert Pidle', value=0.25)
        st.write('Pidle: ', Pidle)

        with open("Varona_k_range.py") as file:
            btn = st.download_button(
                label="Download Python Resources File",
                data=file,
            )

    with col1:
        f, VrectA, VrectB, EffA, I1A, I2A, I1A1, I2A1, I1B, I2B = Varona_k_range.Varona(Vin, Lp, Ls, Rp, Rs, wp, ws, k, Rload, RminLoad, DutyC, Pidle)
        pass

    with col2:
        pass


main()
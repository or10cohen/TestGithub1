import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import Varona_k_range
import CalcVrect

def main():
    epsilon = 000000.1
    st.title('Varona')
    col1, col2, col3 = st.columns([2,5,2])
    with col1:
        Vin = st.number_input('Insert Vin', value=48)
        st.write('Vin:', Vin)

        Lp = st.number_input('Insert Lp', value=18.86 * (10 ** -6))
        st.write('Lp:', Lp)
        Cp = st.number_input('Insert Cp', value=139 * (10 ** -9))
        st.write('Cp:', Cp)
        wp = 1 / np.sqrt(Cp * Lp)
        st.write('wp:', wp)
        FresP = wp / (2 * np.pi)
        st.write('FresP:', FresP)
        Rp = st.number_input('Insert Rp', value=0.1)
        st.write('Rp:', Rp)

        Ls = st.number_input('Insert Ls', value=18.9 * (10 ** -6))
        st.write('Ls:', Ls)
        Cs = st.number_input('Insert Cs', value=103 * (10 ** -9))
        st.write('Cs:', Cs)
        ws = 1 / np.sqrt(Cs * Ls)
        st.write('ws:', ws)
        FresS = ws / (2 * np.pi)
        st.write('FresS:', FresS)
        Rs = st.number_input('Insert Rs', value=0.1 + epsilon)
        st.write('Rs:', Rs)

        DutyC = st.number_input('Insert DutyC', value=0.4)
        st.write('DutyC:', DutyC)

        Vout = st.number_input('Insert Vout', value=0.175)
        st.write('Vout:', Vout)
        Pout = st.number_input('Insert a number', value=60)
        st.write('The current number is ', Pout)
        k = st.number_input('Insert a number', value=0.175 + epsilon)
        st.write('The current number is ',k)
        Rload = (Vout ** 2) / Pout
        st.write('The current number is ', Rload)
        RminLoad = st.number_input('Insert a number', value=570)
        st.write('The current number is ', RminLoad)

    with col3:
        with open("Varona_k_range.py") as file:
            btn = st.download_button(
                label="Download Python Resources File",
                data=file,
            )
    with col2:
        pass


main()
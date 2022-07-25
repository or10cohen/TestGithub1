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
        Vin = st.number_input('Insert a number', value=48)
        st.write('The current number is ', Vin)

        Lp = st.number_input('Insert a number', value=18.86 * (10 ** -6))
        st.write('The current number is ', Lp)
        Cp = st.number_input('Insert a number', value=139 * (10 ** -9))
        st.write('The current number is ', Cp)
        wp = st.number_input('Insert a number', value=1 / np.sqrt(Cp * Lp))
        st.write('The current number is ', wp)
        FresP = st.number_input('Insert a number', value=wp / (2 * np.pi))
        st.write('The current number is ', FresP)
        Rp = st.number_input('Insert a number', value=0.1)
        st.write('The current number is ', Rp)

        Ls = st.number_input('Insert a number', value=18.9 * (10 ** -6))
        st.write('The current number is ', Ls)
        Cs = st.number_input('Insert a number', value=103 * (10 ** -9))
        st.write('The current number is ', Cs)
        ws = st.number_input('Insert a number', value=1 / np.sqrt(Cs * Ls))
        st.write('The current number is ', ws)
        FresS = st.number_input('Insert a number', value=ws / (2 * np.pi))
        st.write('The current number is ', FresS)
        Rs = st.number_input('Insert a number', value=0.1 + epsilon)
        st.write('The current number is ', Rs)

        DutyC = st.number_input('Insert a number', value=0.4)
        st.write('The current number is ', DutyC)

        Vout = st.number_input('Insert a number', value=0.175)
        st.write('The current number is ', Vout)
        Pout = st.number_input('Insert a number', value=60)
        st.write('The current number is ', Pout)
        k = st.number_input('Insert a number', value=0.175 + epsilon)
        st.write('The current number is ',k)
        Rload = st.number_input('Insert a number', value=(Vout ** 2) / Pout)
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
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import Varona_k_range
import CalcVrect
from PIL import Image
import streamlit.components.v1 as components


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

    imageCarouselComponent = components.declare_component("image-carousel-component", path="frontend/public")

    imageUrls = [
        "https://images.unsplash.com/photo-1522093007474-d86e9bf7ba6f?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=764&q=80",
        "https://images.unsplash.com/photo-1610016302534-6f67f1c968d8?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1075&q=80",
        "https://images.unsplash.com/photo-1516550893923-42d28e5677af?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=872&q=80",
        "https://images.unsplash.com/photo-1541343672885-9be56236302a?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=687&q=80",
        "https://images.unsplash.com/photo-1512470876302-972faa2aa9a4?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
        "https://images.unsplash.com/photo-1528728329032-2972f65dfb3f?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
        "https://images.unsplash.com/photo-1557744813-846c28d0d0db?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1118&q=80",
        "https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
        "https://images.unsplash.com/photo-1595867818082-083862f3d630?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
        "https://images.unsplash.com/photo-1622214366189-72b19cc61597?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=687&q=80",
        "https://images.unsplash.com/photo-1558180077-09f158c76707?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=764&q=80",
        "https://images.unsplash.com/photo-1520106212299-d99c443e4568?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=687&q=80",
        "https://images.unsplash.com/photo-1534430480872-3498386e7856?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
        "https://images.unsplash.com/photo-1571317084911-8899d61cc464?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
        "https://images.unsplash.com/photo-1624704765325-fd4868c9702e?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=764&q=80",
    ]
    selectedImageUrl = imageCarouselComponent(imageUrls=imageUrls, height=200)

    if selectedImageUrl is not None:
        st.image(selectedImageUrl)


main()
import pandas as pd
import streamlit as st
from streamlit_folium import folium_static
import folium
import map2

# st.set_page_config(layout="wide")

location = st.sidebar.selectbox(
            'Which DataSet do you want to use?',
            ('Kryot', 'Raanana'))

st.sidebar.selectbox(
    label="What radius do you want to assign?",
    options=("0.1 mile", "1 mile", "2 miles", "3 miles"),
    key="radius"
)



folium_static(map2.mapObj)






import pandas as pd
import streamlit as st

data = pd.DataFrame({
    'awesome cities' : ['Chicago', 'Minneapolis', 'Louisville', 'Topeka'],
    'lat' : [32.82756176005572, 32.8438602958773,  32.840687364169376, 32.84140849497157],
    'lon' : [35.04920899119111, 35.06277023941001, 35.08903442899851, 35.09864746571065]
})

st.map(data)



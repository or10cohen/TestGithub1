import pandas as pd
import streamlit as st
from streamlit_folium import folium_static
import folium
import map2


Raanana  = pd.DataFrame({
    'awesome cities' : ['R1', 'R2', 'R3', 'R4'],
    'lat' : [32.19257001621871, 32.19257001621871, 32.188691433645396, 32.19070709561081],
    'lon' : [34.87963762591485, 34.86982186463729, 34.86978577727965, 34.86058350108193]
})


Kryot = pd.DataFrame({
    'awesome cities' : ['H1', 'H2', 'H3', 'H3'],
    'lat' : [32.83176922748003, 32.83366316684312, 32.8331020038359, 32.834615966742696],
    'lon' : [35.06681117327111, 35.06722858061457, 35.06481457481161, 35.065990272162324]
})
#
location = st.radio(
            'Which DataSet do you want to use?',
            ('Kryot', 'Raanana'))

if location == 'Kryot':
    data = Kryot
elif location == 'Raanana':
    data = Raanana
else:
    pass


st.map(data, zoom=None, use_container_width=True)


st.sidebar.selectbox(
    label="What radius do you want to assign?",
    options=("0.1 mile", "1 mile", "2 miles", "3 miles"),
    key="radius"
)





lat1, lon1 = 32.19257001621871, 34.87963762591485

m = folium.Map(location=[lat1, lon1])
folium.Marker([lat1, lon1]).add_to(m)
layer = folium.FeatureGroup("PP").add_to(m)
# folium.Circle([lat, lon], radius=0.1).add_to(m)  # radius is in meters

# dataDF = pd.read_excel('test.CSVs')
# for itr in range(len(dataDF)):
#     latVal = dataDF.iloc[itr]['lat']
#     lonVal = dataDF.iloc[itr]['lon']
#     nameStr = dataDF.iloc[itr]['name']
#     folium.Circle(location=[latVal, lonVal]).add_to(layer)

folium_static(map2.mapObj)






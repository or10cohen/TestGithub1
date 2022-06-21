import pandas as pd
import folium

# initialize a map with center and zoom

# f = folium.Figure(width=0, height=0)
mapObj = folium.Map(width=800, height=600, location=[32.19257001621871, 34.87963762591485],
                     zoom_start=12, tiles='openstreetmap')
# create a layer for bubble map using FeatureGroup
powerPlantsLayer = folium.FeatureGroup("Power Plants")
# add the created layer to the map
powerPlantsLayer.add_to(mapObj)

# read excel data as dataframe
dataDf = pd.read_excel('power_plants_2.xlsx')
# iterate through each dataframe row
for i in range(len(dataDf)):
    areaStr = dataDf.iloc[i]['area']
    fuelStr = dataDf.iloc[i]['fuel']
    capVal = dataDf.iloc[i]['capacity']


    if fuelStr.lower() == 'wind':
        clr = 'red'
    elif fuelStr.lower() == 'wind':
        clr = 'red'
    else:
        clr = 'red'


    # derive the circle color
    clr = "blue" if fuelStr.lower() == 'wind' else "red"
    # derive the circle pop up html content
    popUpStr = 'Area - {0}<br>Fuel - {1}<br>'.format(
        areaStr, fuelStr)
    # draw a circle for the power plant on the layer
    folium.Circle(
        location=[dataDf.iloc[i]['lat'], dataDf.iloc[i]['lng']],
        popup=folium.Popup(popUpStr, min_width=100, max_width=700),
        radius=0,
        color=clr,
        weight=2,
        fill=True,
        fill_color=clr,
        fill_opacity=0.1
    ).add_to(powerPlantsLayer)


# add layer control over the map
folium.LayerControl().add_to(mapObj)

# html to be injected for displaying legend
legendHtml = '''
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 150px; height: 70px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     ">&nbsp; Fuel Types <br>
     &nbsp; <i class="fa fa-circle"
                  style="color:blue"></i> &nbsp; Wind<br>
     &nbsp; <i class="fa fa-circle"
                  style="color:red"></i> &nbsp; Solar<br>
      </div>
     '''

# inject html corresponding to the legend into the map
mapObj.get_root().html.add_child(folium.Element(legendHtml))

# save the map as html file
mapObj.save('output.html')
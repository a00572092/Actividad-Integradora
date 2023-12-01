#Cambios listados
# 1) Estetico: Cambio de colores, agregado de imagen SFPD como titulo
# 2) Agregado filtro de resolution 
# 3) Gráfica top 5 vecindadas on mayor incidencia de asaltos
# 4) Mapa de calor de San Francisco de incidencias de robo
# 5) Incidencias delictivas por día en la vecindad Mission
# 6) Homicidios cometidos al paso del tiempo en San Francisco


import datetime 
import pickle 
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit_authenticator as stauth
import time
import numpy as np 
from sklearn.impute import SimpleImputer
import pandas as pd 
import matplotlib.pyplot as plt
import base64
import plotly
from streamlit_folium import folium_static
import folium
from folium.plugins import HeatMap
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import hydralit_components as hc
import pickle
from pathlib import Path
from PIL import Image
from streamlit_extras.add_vertical_space import add_vertical_space 
from streamlit_extras.metric_cards import style_metric_cards 
from streamlit_extras.colored_header import colored_header
from streamlit_metrics import metric, metric_row 
from streamlit_card import card
from streamlit_option_menu import option_menu
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
st.set_page_config(page_title='SFPD Dashboard ', page_icon=':bar_chart:', layout='wide')

def logo_to_base64(image):
    import base64
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

logo_path = "rb.jpg"
logo = Image.open(r"C:/Users/mario/Downloads/Seal_of_the_San_Francisco_Police_Department.png")
title = "SFPD Dashboard"
st.markdown(
    """
    <div style="display: flex; align-items: center; background-color: #000000; padding: 1rem; border-radius: 10px;">
        <img src="data:image/png;base64,{}" alt="Logo" style="width: 500px; height:300px; margin-right: 40px;">
        <h1 style="color: white;"> {}</h1>
    </div>
    """.format(logo_to_base64(logo), title),
    unsafe_allow_html=True,
)


data = pd.read_csv("Police.csv")
df = data.fillna(method='bfill')
st.subheader('Police Incident Reports from 2018 to 2020 in San Francisco')
st.sidebar.title('SFPD Category Filter')
st.markdown("The data shown below belongs to incident reports in the city of San Francisco, from the year 2018 t0 2020 which details from each case such as date, day of the week, police district, neighborhood in which it happened, type of incident in category and subcategory, exact location and resolution")
mapa=pd.DataFrame()
mapa['Date'] = df['Incident Date']
mapa['Day'] = df['Incident Date']
mapa['Police District'] = df['Police District']
mapa['Neighborhood'] = df['Analysis Neighborhood']
mapa['Incident Category'] = df['Incident Category']
mapa['Incident Subcategory'] = df['Incident Subcategory']
mapa['Resolution'] = df['Resolution']
mapa['lat'] = df['Latitude']
mapa['lon'] = df['Longitude']
subset_data2=mapa

police_district_input=st.sidebar.multiselect(
    'Police District',
    mapa.groupby('Police District').count().reset_index()['Police District'].tolist())
if len(police_district_input)>0:
    subset_data2 = mapa[mapa['Police District'].isin(police_district_input)]

subset_data1=subset_data2
neighborhood_input = st.sidebar.multiselect(
    'Neighborhood',
    subset_data2.groupby('Neighborhood').count().reset_index()['Neighborhood'].tolist())
if len(neighborhood_input) >0:
    subset_data1 = subset_data2[subset_data2['Neighborhood'].isin(neighborhood_input)]
    
    
subset_data = subset_data1
incident_input = st.sidebar.multiselect(
    'Incident Category',
    subset_data1.groupby('Incident Category').count().reset_index()['Incident Category'].tolist())
if len(incident_input) >0:
    subset_data = subset_data1[subset_data1['Incident Category'].isin(incident_input)]
subset_data0=subset_data

resolution_input=st.sidebar.multiselect(
    'Resolution of crime ',
    mapa.groupby('Resolution').count().reset_index()['Resolution'].tolist())
if len(resolution_input)>0:
    subset_data0 = mapa[mapa['Resolution'].isin(resolution_input)]

st.title("Report your crime!")
st.data_editor(subset_data0, num_rows='dynamic')

st.header('Information display')
st.header('Database')
subset_data0
st.markdown("It is important to mention that any police district can answer to any incident, the neighborhood in which it happened is not related to the police district.")
st.markdown("Crime location in San Francisco")
st.map(subset_data)
st.subheader("Crimes ocurred per day of the week")
st.bar_chart(subset_data0['Day'].value_counts(), color = '#FFC0CB')
st.subheader('Crimes ocurredd per date')
st.line_chart(subset_data0['Date'].value_counts(), color = '#00FF00')
st.subheader('Type of crimes committed')
st.bar_chart(subset_data0['Incident Category'].value_counts(), color = '#FF0000')
agree = st.button('Click to see Incident Subcategories')
if agree:
    st.subheader('Subtype of crimes committed')
    st.bar_chart(subset_data0['Incident Subcategory'].value_counts(), color ='#FFFF00')
st.markdown('Resolution status')
figi, ax1 = plt.subplots()
labels = subset_data0['Resolution'].unique()
ax1.pie(subset_data0['Resolution'].value_counts(), labels = labels, autopct='%1.1f%%', startangle = 20)
st.pyplot(figi)
    
                           
assault_data = subset_data0[subset_data0['Incident Category'] == 'Assault']

top_neighborhoods = assault_data['Neighborhood'].value_counts().nlargest(5).index

st.title('Top 5 Neighborhoods with the Most Assault Incidents')
fig3 = px.scatter_mapbox(
    assault_data[assault_data['Neighborhood'].isin(top_neighborhoods)],
    lat='lat',
    lon='lon',
    color='Neighborhood',
    size_max=15,
    zoom=10,
    mapbox_style="carto-positron",
)

ranking_df = pd.DataFrame({
    'Neighborhood': top_neighborhoods,
    'Rank': range(1, 6),
    'Number of Assault Incidents': assault_data['Neighborhood'].value_counts().nlargest(5).values
})

fig1, table1=st.columns(2)
with fig1:
    st.plotly_chart(fig3)
with table1:
    st.table(ranking_df)

mission_data = subset_data0[subset_data0['Neighborhood'] == 'Mission']

st.subheader('Crime Occurrences per Day in Mission Neighborhood')
line_chart_data = mission_data.groupby('Date').size()
st.line_chart(line_chart_data, color = '#FF0000')

st.title('Crime Heatmap for Frequency of BURGLARY by Neighborhood')

left, cent, right = st.columns(3)

with cent:

    burglary_data = df[df['Incident Category'] == 'Burglary']

    m = folium.Map(location=[burglary_data['Latitude'].mean(), burglary_data['Longitude'].mean()], zoom_start=10)
    heat_data = [[row['Latitude'], row['Longitude']] for index, row in burglary_data.iterrows()]
    HeatMap(heat_data, gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1: 'red'}).add_to(m)

    gradient_legend = """
        <div style="position: fixed; bottom: 50px; left: 50px; width: 200px; height: 150px;
                    background: linear-gradient(to bottom, blue, cyan, lime, yellow, red);
                    border: 2px solid black; z-index:9999; padding: 10px;
                    text-align: center; font-size:14px;">
            <strong>Heatmap Legend</strong><br>
            <span style="color: blue;">0%</span> |
            <span style="color: cyan;">20%</span> |
            <span style="color: lime;">40%</span> |
            <span style="color: yellow;">60%</span> |
            <span style="color: red;">80%+</span>
        </div>
    """
    m.get_root().html.add_child(folium.Element(gradient_legend))

    folium_static(m)



mission_data = subset_data0[subset_data0['Incident Category'] == 'Homicide']
st.subheader('Homicides commited as time passes')
line_chart_data = mission_data.groupby('Date').size()
st.line_chart(line_chart_data)

#

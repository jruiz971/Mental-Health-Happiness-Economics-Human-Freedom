import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from yellowbrick.cluster import silhouette_visualizer
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster import hierarchy
import pydeck as pdk


links = [
 "2015.csv",                   # country
 "2016.csv",                   # country
 "2017.csv",                   # country
 "2018.csv",                   # country
 "2019.csv",                   # country
 ]
rawdata = [ pd.read_csv(l) for l in links ]
rawdata[0]['year']  = 2015
rawdata[1]['year']  = 2016
rawdata[2]['year']  = 2017
rawdata[3]['year']  = 2018
rawdata[4]['year'] = 2019

rawdata[0] = rawdata[0].rename( columns={'Happiness Rank':'Rank', 'Happiness Score':'Score', 'Economy (GDP per Capita)':'GDP per Capita', 'Health (Life Expectancy)':'Healthy', 'Trust (Government Corruption)':'Corruption'})
rawdata[1] = rawdata[1].rename( columns={'Happiness Rank':'Rank', 'Happiness Score':'Score', 'Economy (GDP per Capita)':'GDP per Capita', 'Health (Life Expectancy)':'Healthy', 'Trust (Government Corruption)':'Corruption'})
rawdata[2] = rawdata[2].rename( columns={'Happiness.Rank':'Rank', 'Happiness.Score':'Score', 'Economy..GDP.per.Capita.':'GDP per Capita', 'Health..Life.Expectancy.':'Healthy', 'Trust..Government.Corruption.':'Corruption'})
rawdata[3] = rawdata[3].rename( columns={'Overall rank':'Rank', 'Country or region':'Country', 'GDP per capita':'GDP per Capita', 'Healthy life expectancy':'Healthy', 'Perceptions of corruption':'Corruption', 'Freedom to make life choices':'Freedom'})
rawdata[4] = rawdata[4].rename(columns={'Overall rank':'Rank', 'Country or region':'Country', 'GDP per capita':'GDP per Capita', 'Healthy life expectancy':'Healthy', 'Perceptions of corruption':'Corruption', 'Freedom to make life choices':'Freedom'})

WHR = pd.concat(rawdata)[['Rank', 'Country', 'year', 'Score', 'GDP per Capita', 'Healthy', 'Freedom', 'Corruption']]
WHR = WHR.dropna()
st.title('Exploracion de datos sobre reportes de la felicidad en el mundo.')
st.write('Datos obtenidos de Kaggle de World Happiness Report 2015-2019')
st.write(WHR)

st.write('Aquí tenemos un primer análisis exploratorio de los datos')

def hexbin(x, y, color, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    plt.hexbin(x, y, gridsize=15, cmap=cmap, **kwargs)

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_,counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)

def first(WHR):
    g = sns.pairplot(WHR)
    g.map_lower(hexbin);
    g.map_upper(sns.kdeplot)
    st.pyplot(g)

def clusters(WHR):
    st.subheader('Healty - Corruption')
    fig, ax = plt.subplots()
    kmeans = KMeans()
    visualizer = KElbowVisualizer(kmeans, k=10)
    HC = ['Healthy','Corruption']
    visualizer.fit(WHR[HC])
    st.pyplot(plt)

    kmeans = KMeans(n_clusters=4).fit(WHR[HC])
    labels = kmeans.predict(WHR[HC])
    plt.style.use('classic')
    fig, ax = plt.subplots()
    ax.set_xlabel('Healthy')
    ax.set_ylabel('Corruption')
    ax.scatter(WHR['Healthy'], WHR['Corruption'], c=labels,s=60)
    st.pyplot(fig)

    st.subheader('Healty - GDP')
    fig, ax = plt.subplots()
    kmeans = KMeans()
    visualizer = KElbowVisualizer(kmeans, k=10)
    HG = ['Healthy','GDP per Capita']
    visualizer.fit(WHR[HC])
    st.pyplot(plt)

    kmeans = KMeans(n_clusters=3).fit(WHR[HC])
    labels = kmeans.predict(WHR[HG])
    plt.style.use('classic')
    fig, ax = plt.subplots()
    ax.set_xlabel('Healthy')
    ax.set_ylabel('GDP per Capita')
    ax.scatter(WHR['Healthy'], WHR['GDP per Capita'], c=labels,s=60)
    st.pyplot(fig)

    st.subheader('GDP per Capita - Corruption')
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    CG = ['GDP per Capita','Corruption']
    model = model.fit(WHR[CG])
    plot_dendrogram(model, truncate_mode='level', p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    st.pyplot(plt)

    clustering = AgglomerativeClustering(n_clusters=4).fit(WHR[CG])
    labels = clustering.labels_
    plt.style.use('classic')
    fig, ax = plt.subplots()
    ax.set_xlabel('GDP per Capita')
    ax.set_ylabel('Corruption')
    ax.scatter(WHR['GDP per Capita'], WHR['Corruption'], c=labels, s=60)
    st.pyplot(fig)

posicion = pd.read_csv('Posicion_pais.csv', index_col = 'Country/Region')

def show_map(df_,coords):
    left_col , right_col = st.beta_columns(2)

    with left_col:
        start_, end_ = st.select_slider(
            'Año:',
            options=list([2015,2016,2017,2018,2019]),
            value= (2015,2019)
        )

    with right_col:

        variable = st.selectbox("Elige una variable", list(['Rank','GDP per Capita','Healthy','Freedom','Corruption']))

    WHR = df_.reset_index()
    Def = WHR.sort_values(by=['Rank']).head(10).Country.unique()
    Def=list(Def)
    Options = list(WHR.Country.unique())
    country = st.multiselect("Country", Options,Def)
    st.write('Comportamiento de '+variable+' del año '+str(start_)+' al año '+str(end_))

    datos = df_.loc[:, ['Country', 'year',variable]]
    datos = datos[(datos['year'] >= start_) & (datos['year'] <= end_)]

    valid_keys = pd.DataFrame([key  if key in coords.index.values else pd.NA for key in datos['Country']])
    datos['coordsKeys'] = valid_keys
    datos = datos.dropna()
    coord_datos = coords.loc[datos['coordsKeys']]
    datos = datos.drop(columns=['coordsKeys'])
    datos = datos.reset_index()
    datos = datos[datos['Country'].isin(country)]
    g = sns.relplot(data=datos, kind="line", x="year", y=variable, hue='Country')
    st.pyplot(g)
    cols =['Rank', 'year', 'Score', 'GDP per Capita', 'Healthy', 'Freedom', 'Corruption']
    st_ms = st.multiselect("Columnas",cols, ['GDP per Capita','Healthy','Freedom'])
    st_ms.append('Country')
    datos2 = df_[st_ms].groupby('Country').mean()
    print(datos2.index)
    datos2 = datos2[datos2.index.isin(country)]
    datos2 = datos2.reset_index()
    g = sns.pairplot(datos2,hue='Country')
    st.pyplot(g)

#first(WHR)
#clusters(WHR)
show_map(WHR, posicion)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 1 2020
@authors: Juan Luis Ruiz Vanegas

Contact info: juanluisruiz971@gmail.com

This project is under GPL-3.0 License 
"""
import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("Suicide rates overview 1985-2016")
st.sidebar.title("Suicide rates overview 1985-2016")
st.markdown(" This application is a Streamlit dashboard to analyze the the suicidal rates over differents countries filtered by age and socio-economic spectrum")


#-Load Dataset-#
@st.cache(persist=True) #To prevent every time it recharges from doing the function again
#Unless there is a change in the function (ie, parameters)
def load_data():
    """
    [summary] Load the csv data
	Returns:
	[Dataframe]: [Pandas Dataframe with the csv information]
	"""
    data = pd.read_csv("../datasets/Suicide Rates Overview 1985 to 2016.csv")
    return data

data = load_data()
st.write(data) #Interactive table. Showing total data

st.sidebar.subheader("Select the year, country and sex you want to know the suicide rate")

year = st.sidebar.slider("Year", 1985,2016)
sex = st.sidebar.multiselect("Sex", ('male', 'female'),key=0)
country = st.sidebar.multiselect("Country",('Albania', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Aruba',
       'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain',
       'Barbados', 'Belarus', 'Belgium', 'Belize',
       'Bosnia and Herzegovina', 'Brazil', 'Bulgaria', 'Cabo Verde',
       'Canada', 'Chile', 'Colombia', 'Costa Rica', 'Croatia', 'Cuba',
       'Cyprus', 'Czech Republic', 'Denmark', 'Dominica', 'Ecuador',
       'El Salvador', 'Estonia', 'Fiji', 'Finland', 'France', 'Georgia',
       'Germany', 'Greece', 'Grenada', 'Guatemala', 'Guyana', 'Hungary',
       'Iceland', 'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan',
       'Kazakhstan', 'Kiribati', 'Kuwait', 'Kyrgyzstan', 'Latvia',
       'Lithuania', 'Luxembourg', 'Macau', 'Maldives', 'Malta',
       'Mauritius', 'Mexico', 'Mongolia', 'Montenegro', 'Netherlands',
       'New Zealand', 'Nicaragua', 'Norway', 'Oman', 'Panama', 'Paraguay',
       'Philippines', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar',
       'Republic of Korea', 'Romania', 'Russian Federation',
       'Saint Kitts and Nevis', 'Saint Lucia',
       'Saint Vincent and Grenadines', 'San Marino', 'Serbia',
       'Seychelles', 'Singapore', 'Slovakia', 'Slovenia', 'South Africa',
       'Spain', 'Sri Lanka', 'Suriname', 'Sweden', 'Switzerland',
       'Thailand', 'Trinidad and Tobago', 'Turkey', 'Turkmenistan',
       'Ukraine', 'United Arab Emirates', 'United Kingdom',
       'United States', 'Uruguay', 'Uzbekistan'),key=0)

###Create dataframe made of queries
if (len (country)>0 and len(sex)>0): #Both arrays
    df = pd.DataFrame() #Empty Dataframe
    for s in sex: #['male'] or ['male','female']
        for c in country: #Array with some contries []
            aux_df = data.loc[(data['year']==year) & (data['sex']==s) & (data['country']==c)] #query
            df = df.append(aux_df) #Creating dataframe made of queries

    if not st.sidebar.checkbox ("Close", True, key='1'): #Checkbox to show/hide Dataframe on web service
        st.markdown("### Query Results")
        st.write(df) #Show Dataframe on web service
else:
    st.markdown('### Please, selact at least one country and one sex')

###Plotting the dataframe made of the queries
st.markdown("### Queried Yearly Suicide Rates Overview Graph")
#st.sidebar.subheader("Queried Yearly Suicide Rates Overview Graph")
fig_choice = px.histogram(df,x='age', y="suicides/100k pop", color="gdp_per_capita ($)",
    facet_col="country", labels={'gdp_per_capita ($)'},height=600,width=800)
st.plotly_chart(fig_choice)
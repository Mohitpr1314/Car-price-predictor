import streamlit as st
import pickle
import pandas as pd
import numpy as np

pipe = pickle.load(open('LRModel.pkl','rb'))
df = pickle.load(open('Cleaned_car.pkl','rb'))


st.title('Car Price Predictor')

# name of car
name = st.selectbox('Car Name', df['name'].unique())

# company
company = st.selectbox('Company', df['company'].unique())

# year
year = st.number_input("Enter an year:", step=1, value=2000, format="%d")


# km
km = st.number_input("kms_driven", step=1, value=0, format="%d")


# fuel
fuel = st.selectbox('Fuel Type', df['fuel_type'].unique())

if st.button('Price Predict'):
    query_df = pd.DataFrame({
        'name': [name],
        'company': [company],
        'year': [year],
        'kms_driven': [km],
        'fuel_type': [fuel]
    })
    st.subheader("The predicted price of this configuration is : " + str(int(pipe.predict(query_df)[0])))


st.image('car5.jpg')

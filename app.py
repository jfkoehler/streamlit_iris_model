import streamlit as st 
import pandas as pd
import numpy as np
import pickle

st.header('Iris Classifier')
st.write('Please enter the measurements of the flower you are trying to classify.')

sl = st.number_input('Sepal Length')
sw = st.number_input('Sepal Width')
pl = st.number_input('Petal Length')
pw = st.number_input('Petal Width')

X = np.array([[sl, sw, pl, pw]])

with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

prediction = model.predict(X)

iris_dict = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

st.write(iris_dict[prediction[0]])
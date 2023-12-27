import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


st.write("### Simple Iris Prediction app\n #### This application will predict the flowers' type")


st.sidebar.header("Information")

def user_input_features():
    sepal_lenght = st.sidebar.slider('Sepal Lenght', 4.30,5.40,7.90)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0,3.40,4.40)
    petal_lenght = st.sidebar.slider('Petal Lenght', 1.00,1.30,6.90)
    petal_width = st.sidebar.slider('Petal Width', 0.10,0.20,2.50)
    data_dic = {
        'sepal_lenght':sepal_lenght,
        'sepal_width' : sepal_width,
        'petal_lenght' : petal_lenght,
        'petal_width' : petal_width
    }
    return pd.DataFrame(data_dic, index = [0])

df = user_input_features()

st.write("User inputs:")
st.write(df)

iris = datasets.load_iris()

X = iris.data
y = iris.target

model = RandomForestClassifier().fit(X,y)

prediction = model.predict(df)
prediction_prob = model.predict_proba(df)

st.subheader("Labels")
st.write(iris.target_names)

st.subheader("Prediction Probability")

table_data = pd.DataFrame({'Target Names': iris.target_names, 'Prob.': prediction_prob[0]})

st.table(table_data)

st.subheader("Prediction")
st.write(iris.target_names[prediction])
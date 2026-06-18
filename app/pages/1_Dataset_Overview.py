import streamlit as st
import pandas as pd

st.title("Dataset Overview")

df = pd.read_csv(
    r"data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

st.write(df.head())

st.write("Shape:", df.shape)

st.write(df.describe())
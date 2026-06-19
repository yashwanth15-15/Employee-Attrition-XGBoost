import streamlit as st
import pandas as pd
import plotly.express as px

features = {
    "Feature":[
        "OverTime",
        "StockOptionLevel",
        "TotalWorkingYears",
        "JobInvolvement",
        "MonthlyIncome",
        "JobLevel",
        "Age",
        "NumCompaniesWorked",
        "MaritalStatus",
        "YearsAtCompany"
    ],
    "Score":[100,92,85,80,75,70,65,60,55,50]
}

df = pd.DataFrame(features)

fig = px.bar(
    df,
    x="Score",
    y="Feature",
    orientation="h",
    title="Top Features Affecting Attrition"
)

st.plotly_chart(fig, width="stretch")
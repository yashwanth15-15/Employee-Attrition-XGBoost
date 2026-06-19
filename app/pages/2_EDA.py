import streamlit as st

st.title("📊 Exploratory Data Analysis")

st.write("""
Exploratory Data Analysis performed on the IBM HR Employee Attrition Dataset.
""")

st.subheader("Employee Attrition Distribution")
st.image("screenshots/attrition_distribution.png")

st.subheader("OverTime vs Attrition")
st.image("screenshots/OVERTIME  VS ATTRITION.png")

st.subheader("Job Satisfaction vs Attrition")
st.image("screenshots/JOB SATISFACTION VD ATTRITION.png")

st.subheader("Department vs Attrition")
st.image("screenshots/DEPARTMRNT VS ATTRITION.png")

st.subheader("Monthly Income Distribution")
st.image("screenshots/MONTHLY INCOME DISTRIBUTION.png")

st.success(
    "EDA completed successfully. These insights helped identify important factors influencing employee attrition."
)
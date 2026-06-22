import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Dataset Overview")

# Load Dataset
df = pd.read_csv(
    "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# ==========================
# KPI CARDS
# ==========================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Employees",
        df.shape[0]
    )

with col2:
    st.metric(
        "Features",
        df.shape[1]
    )

with col3:
    st.metric(
        "Attrition Cases",
        (df["Attrition"] == "Yes").sum()
    )

st.markdown("---")

# ==========================
# DATASET PREVIEW
# ==========================

st.subheader("📄 Dataset Preview")

st.dataframe(
    df.head(),
    width="stretch"
)

# ==========================
# DATA QUALITY
# ==========================

st.subheader("🔍 Data Quality")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

with col2:
    st.metric(
        "Duplicate Records",
        int(df.duplicated().sum())
    )

st.markdown("---")

# ==========================
# DATASET INFORMATION
# ==========================

st.subheader("ℹ️ Dataset Information")

st.info("""
Dataset Source: IBM HR Analytics Employee Attrition Dataset

Target Variable: Attrition

Total Employees: 1470

Total Features: 35
""")

# ==========================
# STATISTICAL SUMMARY
# ==========================

st.subheader("📈 Statistical Summary")

with st.expander("View Statistical Summary"):
    st.dataframe(
        df.describe(),
        width="stretch"
    )
# ==========================
# ATTRITION DISTRIBUTION
# ==========================

st.subheader("🎯 Attrition Distribution")

attrition_counts = (
    df["Attrition"]
    .value_counts()
    .reset_index()
)

attrition_counts.columns = [
    "Attrition",
    "Count"
]

fig = px.bar(
    attrition_counts,
    x="Attrition",
    y="Count",
    text="Count",
    title="Employee Attrition Distribution"
)

fig.update_traces(
    textposition="outside"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==========================
# COLUMN LIST
# ==========================

st.subheader("🧾 Dataset Columns")

with st.expander("View All Dataset Columns"):
    st.write(list(df.columns))

# ==========================
# TARGET VARIABLE SUMMARY
# ==========================

st.subheader("🎯 Target Variable Summary")

yes_count = (
    df["Attrition"] == "Yes"
).sum()

no_count = (
    df["Attrition"] == "No"
).sum()

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Employees Left",
        yes_count
    )

with col2:
    st.metric(
        "Employees Stayed",
        no_count
    )

st.markdown("---")

st.caption(
    "IBM HR Analytics Employee Attrition Dataset"
)
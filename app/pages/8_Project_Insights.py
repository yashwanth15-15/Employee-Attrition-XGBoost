import streamlit as st
import pandas as pd
import pickle

st.title("📊 Project Insights")

st.markdown("""
## Employee Attrition Prediction Using XGBoost

This page provides an overview of the project,
dataset, model performance, and business insights.
""")

st.markdown("---")

# Project Metrics

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Dataset Size",
        "1,470"
    )

with col2:
    st.metric(
        "Features",
        "34"
    )

with col3:
    st.metric(
        "Algorithm",
        "XGBoost"
    )

with col4:
    st.metric(
        "Accuracy",
        "85.03%"
    )

st.markdown("---")

st.subheader("🎯 Project Objective")

st.write("""
The objective of this project is to predict employee attrition
before employees leave the organization, enabling HR teams
to take proactive retention measures.
""")

st.markdown("---")

st.subheader("🛠 Technology Stack")

tech_df = pd.DataFrame({
    "Technology": [
        "Python",
        "Pandas",
        "Scikit-Learn",
        "XGBoost",
        "Streamlit",
        "Plotly",
        "Git",
        "GitHub"
    ]
})

st.dataframe(
    tech_df,
    width="stretch"
)

st.markdown("---")

st.subheader("📈 Model Performance")

performance_df = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost"
    ],
    "Accuracy (%)": [
        86.05,
        75.85,
        87.76,
        85.71
    ]
})

st.dataframe(
    performance_df,
    width="stretch"
)

st.success(
    "XGBoost selected as final model after tuning and evaluation."
)

st.markdown("---")

st.subheader("🔥 Top Features Affecting Attrition")

top_features = pd.DataFrame({
    "Feature": [
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
    ]
})

st.dataframe(
    top_features,
    width="stretch"
)

st.markdown("---")

st.subheader("💡 Business Impact")

st.write("""
✅ Early identification of employees at risk

✅ Supports HR decision making

✅ Improves employee retention

✅ Reduces recruitment costs

✅ Enhances workforce planning

✅ Provides actionable HR insights
""")

st.markdown("---")

st.subheader("🚀 Future Enhancements")

st.write("""
• SHAP Explainability Dashboard

• Email Alerts for High Risk Employees

• Database Integration

• Employee Login System

• HR Admin Dashboard

• Automated Monthly Reports
""")

st.markdown("---")

st.subheader("👨‍💻 Developer")

st.info("""
B. Yashwanth

B.Tech Computer Science & Engineering

Employee Attrition Prediction Using XGBoost

Major Project - 2026
""")

st.caption(
    "Employee Attrition Prediction System | Project Insights"
)

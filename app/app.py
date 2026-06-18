import streamlit as st

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
# 🚀 Employee Attrition Prediction Using XGBoost

### AI-Powered HR Analytics Dashboard
""")

st.markdown("---")

st.markdown("""
## 📌 Major Project

**Early Prediction of Employee Attrition Using XGBoost Machine Learning Model**

This project helps Human Resource departments identify employees who are at risk of leaving the organization by using Machine Learning and predictive analytics.

### 🎯 Objectives

- Predict employee attrition risk
- Analyze employee data
- Identify key attrition factors
- Support HR decision-making
- Improve employee retention

### ⚙️ Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-Learn
- XGBoost
- Plotly
- SHAP

### 📊 Dashboard Modules

✅ Dataset Overview

✅ Exploratory Data Analysis (EDA)

✅ Model Performance Evaluation

✅ Employee Prediction

✅ Feature Importance Analysis

✅ Batch Employee Prediction

✅ HR Insights & Recommendations

### 🤖 Machine Learning Model

**Algorithm:** XGBoost Classifier

**Model Accuracy:** 85.03%

**Best Tuned Score:** 88.03%

**Target Variable:** Employee Attrition

### 🔍 Key Features Affecting Attrition

1. OverTime
2. StockOptionLevel
3. TotalWorkingYears
4. JobInvolvement
5. MonthlyIncome
6. JobLevel
7. Age
8. NumCompaniesWorked
9. MaritalStatus
10. YearsAtCompany

### 💡 Business Impact

This system enables organizations to:

- Reduce employee turnover
- Improve workforce planning
- Increase employee satisfaction
- Support retention strategies
- Make data-driven HR decisions
""")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model Accuracy",
        "85.03%"
    )

with col2:
    st.metric(
        "Best Tuned Score",
        "88.03%"
    )

with col3:
    st.metric(
        "Project Type",
        "Major Project"
    )

st.markdown("---")

st.info(
    "Use the navigation menu on the left to explore all dashboard modules."
)

st.caption(
    "Employee Attrition Prediction System | XGBoost Machine Learning Model | Developed by B. Yashwanth"
)
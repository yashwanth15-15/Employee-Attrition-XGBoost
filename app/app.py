import streamlit as st

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("🚀 Employee Attrition Prediction System")

st.markdown(
    "### Predict Employee Turnover Using Machine Learning & HR Analytics"
)

st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model Accuracy", "85.03%")

with col2:
    st.metric("Dataset Size", "10,000")

with col3:
    st.metric("Features", "34")

st.markdown("""
## 📌 

**Early Prediction of Employee Attrition Using XGBoost Machine Learning Model**

This project helps Human Resource departments identify employees who are at risk of leaving the organization by using Machine Learning and predictive analytics.

### 🎯 Objectives

🎯 Project Objectives

• Predict employee attrition before resignation

• Identify employees at retention risk

• Analyze key workforce trends

• Support HR decision-making

• Reduce employee turnover costs

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
            """)
st.success("""
✅ Reduce employee turnover

✅ Improve workforce planning

✅ Increase employee satisfaction

✅ Support retention strategies

✅ Enable data-driven HR decisions

""")

st.markdown("---")

col1, col2, = st.columns(2)

with col1:
    st.metric(
        "Best Tuned Score",
        "88.03%"
    )
with col2:
    st.metric(
        "Prediction Model",
        "XGBoost"
    )


st.markdown("---")

st.info(
    "Use the navigation menu on the left to explore all dashboard modules."
)

st.caption(
    "Employee Attrition Prediction System | XGBoost Machine Learning Model | Developed by B. Yashwanth"
)
st.info("""
📌 Dashboard Navigation

1. Dataset Overview
2. Exploratory Data Analysis
3. Model Performance
4. Employee Prediction
5. Feature Importance
6. Batch Prediction
7. HR Insights
8. About Project
9. SHAP Explainability
""")
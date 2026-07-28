import streamlit as st

st.title("ℹ️ About Project")

st.info(
    """
    Early Prediction of Employee Attrition Using XGBoost Machine Learning Model
    """
)

st.subheader("🎯 Project Objective")

st.write("""
The objective of this project is to predict employee attrition
using Machine Learning techniques and assist Human Resource
departments in identifying employees who are at risk of leaving
the organization.
""")

st.subheader("🚀 Project Goals")

st.success("""
✅ Predict employee attrition risk

✅ Support HR decision-making

✅ Improve employee retention

✅ Reduce hiring and replacement costs

✅ Enable data-driven workforce planning
""")

st.subheader("⚙️ Technologies Used")

st.write("""
• Python

• Streamlit

• Pandas

• Scikit-Learn

• XGBoost

• Plotly

• SHAP
""")

st.subheader("📊 Dataset Information")

st.info("""
Dataset: IBM HR Analytics Employee Attrition Dataset

Employees: 1470

Features: 35

Target Variable: Attrition
""")

from helpers.metrics_calculator import get_model_metrics
acc, f1 = get_model_metrics()

st.subheader("🤖 Machine Learning Model")

st.write(f"""
Final Model: XGBoost Classifier

Test Accuracy: {acc*100:.2f}%

Test F1 Score: {f1:.4f}

SMOTE Applied for Class Balancing
""")

st.subheader("💼 Business Benefits")

st.success("""
• Identify high-risk employees

• Improve retention strategies

• Reduce employee turnover

• Increase workforce stability

• Support proactive HR interventions
""")

st.subheader("👨‍💻 Developer")

st.write("""
Name: B. Yashwanth

Degree: B.Tech Computer Science & Engineering

Project Type: Machine Learning Based HR Analytics System
""")

st.markdown("---")

st.caption(
    "Employee Attrition Prediction System | XGBoost Machine Learning Model"
)
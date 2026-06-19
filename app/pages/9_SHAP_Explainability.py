import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

st.title("🔍 SHAP Explainability Dashboard")

st.write("""
Understand why employees are predicted to leave by
analyzing feature importance using SHAP values.
""")

# Load Model
with open("models/final_xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load Encoders
with open("models/final_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Load Feature Names
with open("models/final_features.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Load Dataset
df = pd.read_csv(
    "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# Encode categorical columns
for col, encoder in encoders.items():

    if col == "Attrition":
        continue

    if col in df.columns:

        try:
            df[col] = encoder.transform(
                df[col]
            )
        except:
            pass

# Features
X = df[feature_names]

st.success(
    f"Dataset Loaded Successfully ({len(X)} Employees)"
)

# Create SHAP Explainer
explainer = shap.TreeExplainer(model)

# SHAP Values
shap_values = explainer.shap_values(X)

st.markdown("---")

st.subheader("🌍 Global Feature Importance")

fig, ax = plt.subplots(figsize=(10,6))

shap.summary_plot(
    shap_values,
    X,
    show=False
)

st.pyplot(fig)

st.markdown("---")

st.subheader("📊 Top 10 Important Features")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance":
    abs(shap_values).mean(axis=0)
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

st.dataframe(
    importance.head(10),
    width="stretch"
)

st.markdown("---")

st.subheader("👤 Individual Employee Explanation")

employee_index = st.number_input(
    "Select Employee Index",
    min_value=0,
    max_value=len(X)-1,
    value=0
)

employee_data = X.iloc[
    employee_index
]

prediction = (
    model.predict_proba(
        [employee_data]
    )[0][1]
    * 100
)

st.metric(
    "Predicted Attrition Risk",
    f"{prediction:.2f}%"
)

st.markdown("---")

st.subheader(
    "Top Risk Factors For Selected Employee"
)

employee_shap = pd.DataFrame({
    "Feature": X.columns,
    "SHAP Value":
    shap_values[employee_index]
})

employee_shap = employee_shap.sort_values(
    by="SHAP Value",
    ascending=False
)

st.dataframe(
    employee_shap.head(10),
    width="stretch"
)

st.info("""
Positive SHAP Values increase attrition risk.

Negative SHAP Values reduce attrition risk.

Higher absolute SHAP values indicate stronger influence on prediction.
""")

st.markdown("---")

st.success(
    "Explainable AI successfully integrated into the Employee Attrition Prediction System."
)
import streamlit as st

st.title("Model Performance")

st.subheader("Model Comparison")

st.table({
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

st.subheader("XGBoost Results")

st.write("Accuracy: 85.71%")
st.write("Recall after SMOTE: 36%")
st.write("Best Tuned F1 Score: 0.8803")

st.success("Hyperparameter Tuning Completed")
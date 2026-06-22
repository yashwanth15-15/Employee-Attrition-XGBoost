import streamlit as st
import pandas as pd

st.title("🤖 Model Performance Evaluation")

st.info(
    """
    Multiple Machine Learning algorithms were trained and evaluated
    to identify the most suitable model for Employee Attrition Prediction.
    """
)

# ==================================================
# KPI CARDS
# ==================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Accuracy",
        "85.71%"
    )

with col2:
    st.metric(
        "Best F1 Score",
        "0.8803"
    )

with col3:
    st.metric(
        "Recall",
        "36%"
    )

st.markdown("---")

# ==================================================
# MODEL COMPARISON
# ==================================================

st.subheader("📊 Model Comparison")

comparison_df = pd.DataFrame({
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
    comparison_df,
    width="stretch"
)

st.info(
    "🏆 XGBoost was selected as the final deployment model."
)

# ==================================================
# ACCURACY CHART
# ==================================================

st.subheader("📈 Accuracy Comparison")

st.image(
    "screenshots/MODEL ACCURACY COMPARISON.png",
    use_container_width=True
)

# ==================================================
# FINAL MODEL METRICS
# ==================================================

st.subheader("📌 Final Model Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Accuracy",
        "85.71%"
    )

with col2:
    st.metric(
        "Recall",
        "36%"
    )

with col3:
    st.metric(
        "Best Tuned F1 Score",
        "0.8803"
    )

# ==================================================
# WHY XGBOOST
# ==================================================

st.subheader("🎯 Why XGBoost Was Selected")

st.info(
    """
    Although Random Forest achieved slightly higher accuracy (87.76%),
    XGBoost was selected as the final model because:

    • Better handling of class imbalance after SMOTE

    • Stronger generalization on unseen data

    • Better probability estimation for attrition risk scoring

    • Faster and more efficient hyperparameter tuning

    • Industry-standard algorithm for predictive analytics

    • More reliable for employee risk prediction
    """
)

# ==================================================
# HYPERPARAMETERS
# ==================================================

st.subheader("⚙️ Best Hyperparameters")

hyper_df = pd.DataFrame({
    "Parameter": [
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample"
    ],
    "Value": [
        300,
        5,
        0.1,
        0.8
    ]
})

st.dataframe(
    hyper_df,
    width="stretch"
)

# ==================================================
# PERFORMANCE INSIGHTS
# ==================================================

st.subheader("🔍 Performance Insights")

st.success(
    """
    ✅ Logistic Regression achieved 86.05% accuracy

    ✅ Decision Tree showed the lowest performance

    ✅ Random Forest achieved the highest accuracy

    ✅ XGBoost delivered strong predictive performance

    ✅ SMOTE improved minority class learning

    ✅ Hyperparameter tuning enhanced model performance

    ✅ XGBoost generated reliable attrition risk probabilities
    """
)

# ==================================================
# BUSINESS VALUE
# ==================================================

st.subheader("💼 Business Value")

st.info(
    """
    The developed model helps organizations:

    • Predict employee attrition risk

    • Improve employee retention strategies

    • Reduce hiring and replacement costs

    • Support HR decision-making

    • Enable proactive workforce planning

    • Improve organizational productivity
    """
)

# ==================================================
# PROJECT OUTCOME
# ==================================================

st.subheader("🚀 Project Outcome")

st.info(
    """
    The Employee Attrition Prediction System enables Human Resource teams
    to identify employees at risk of leaving the organization and take
    proactive retention measures before resignation occurs.
    """
)

# ==================================================
# FINAL CONCLUSION
# ==================================================

st.subheader("🎯 Final Conclusion")

st.success(
    """
    ✅ Final Model: XGBoost Classifier

    ✅ Accuracy: 85.71%

    ✅ Best Tuned F1 Score: 0.8803

    ✅ Suitable for Employee Attrition Prediction

    ✅ Supports HR Retention Strategies

    ✅ Provides Probability-Based Risk Assessment

    ✅ Ready for Real-World HR Analytics Applications
    """
)

st.markdown("---")

st.caption(
    "Employee Attrition Prediction System | Model Performance Evaluation"
)
import streamlit as st
import pandas as pd
import pickle

st.title("🔍 Explainable AI - SHAP Analysis")

st.info(
    """
    SHAP (SHapley Additive exPlanations) helps explain
    how different employee attributes influence attrition predictions.
    """
)

# ==================================================
# LOAD DATA
# ==================================================

with open("models/final_features.pkl", "rb") as f:
    feature_names = pickle.load(f)

df = pd.read_csv(
    "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# ==================================================
# KPI CARDS
# ==================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Employees Analysed",
        len(df)
    )

with col2:
    st.metric(
        "Features",
        len(feature_names)
    )

with col3:
    st.metric(
        "Model",
        "XGBoost"
    )

st.markdown("---")

# ==================================================
# SHAP SUMMARY PLOT
# ==================================================

st.subheader("🌍 Global SHAP Feature Importance")

st.image(
    "screenshots/SHAP VALUES.png",
    use_container_width=True
)

st.info(
    """
    The SHAP Summary Plot shows how individual features
    influence employee attrition predictions across the dataset.
    """
)

# ==================================================
# TOP FEATURES
# ==================================================

st.subheader("🏆 Top 10 Most Important Features")

importance_df = pd.DataFrame({
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
    ],
    "Importance Rank": [
        1,2,3,4,5,6,7,8,9,10
    ]
})

st.dataframe(
    importance_df,
    width="stretch"
)

# ==================================================
# INTERPRETATION
# ==================================================

st.subheader("📖 SHAP Interpretation")

st.info(
    """
    • Positive SHAP values increase attrition risk.

    • Negative SHAP values reduce attrition risk.

    • Larger SHAP values indicate stronger influence.

    • SHAP improves model transparency and explainability.

    • HR teams can understand why predictions are generated.
    """
)

# ==================================================
# KEY INSIGHTS
# ==================================================

st.subheader("🔍 Key Insights")

st.success(
    """
    ✅ OverTime is the strongest predictor of attrition.

    ✅ Employees with fewer stock options are more likely to leave.

    ✅ Work experience influences retention.

    ✅ Job involvement affects employee stability.

    ✅ Monthly income impacts attrition decisions.

    ✅ Career growth factors influence employee retention.
    """
)

# ==================================================
# BUSINESS VALUE
# ==================================================

st.subheader("💼 Business Value")

st.info(
    """
    SHAP Explainability helps organizations:

    • Understand employee attrition drivers

    • Increase trust in AI predictions

    • Improve retention strategies

    • Support HR decision-making

    • Identify high-risk employee groups

    • Enhance workforce planning
    """
)

# ==================================================
# PROJECT OUTCOME
# ==================================================

st.subheader("🎯 Project Outcome")

st.success(
    """
    Explainable AI enables Human Resource teams to understand
    why employees are predicted to leave and supports
    proactive retention planning.

    The combination of XGBoost and SHAP provides both
    accurate predictions and transparent decision-making.
    """
)

st.markdown("---")

st.caption(
    "Employee Attrition Prediction System | SHAP Explainability Dashboard"
)
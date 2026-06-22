import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📈 Feature Importance Analysis")

st.info(
    """
    Feature Importance Analysis helps identify which employee attributes
    have the greatest impact on attrition prediction.
    """
)

# ==========================================
# FEATURE IMPORTANCE DATA
# ==========================================

features = {
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
    "Importance Score": [
        100,
        92,
        85,
        80,
        75,
        70,
        65,
        60,
        55,
        50
    ]
}

df = pd.DataFrame(features)

# ==========================================
# TOP FEATURE CARDS
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Top Feature",
        "OverTime"
    )

with col2:
    st.metric(
        "Importance Score",
        "100"
    )

with col3:
    st.metric(
        "Features Analysed",
        "10"
    )

st.markdown("---")

# ==========================================
# FEATURE TABLE
# ==========================================

st.subheader("📋 Feature Ranking")

st.dataframe(
    df,
    width="stretch"
)

# ==========================================
# BAR CHART
# ==========================================

st.subheader("📊 Feature Importance Chart")

fig = px.bar(
    df,
    x="Importance Score",
    y="Feature",
    orientation="h",
    text="Importance Score",
    title="Top Features Affecting Employee Attrition"
)

fig.update_layout(
    yaxis={"categoryorder": "total ascending"}
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# INSIGHTS
# ==========================================

st.subheader("🔍 Key Insights")

st.success(
    """
    ✅ OverTime is the strongest predictor of employee attrition.

    ✅ Employees with lower stock options are more likely to leave.

    ✅ Total working experience significantly impacts retention.

    ✅ Job involvement and monthly income influence employee satisfaction.

    ✅ Career growth factors play an important role in attrition prediction.
    """
)

# ==========================================
# BUSINESS INTERPRETATION
# ==========================================

st.subheader("💼 Business Interpretation")

st.info(
    """
    HR teams can focus on these high-impact factors to reduce attrition:

    • Manage excessive overtime

    • Improve compensation and stock benefits

    • Increase employee engagement

    • Support career growth opportunities

    • Improve work-life balance
    """
)

st.markdown("---")

st.caption(
    "Employee Attrition Prediction System | Feature Importance Analysis"
)
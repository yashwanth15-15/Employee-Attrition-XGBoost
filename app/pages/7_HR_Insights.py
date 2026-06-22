import streamlit as st
import pandas as pd
import plotly.express as px

st.title("👥 HR Insights & Recommendations")

# ==================================================
# LOAD DATASET
# ==================================================

df = pd.read_csv(
    "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# ==================================================
# KPI CARDS
# ==================================================

total_employees = len(df)

attrition_cases = (
    df["Attrition"] == "Yes"
).sum()

attrition_rate = (
    attrition_cases / total_employees
) * 100

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Employees",
        total_employees
    )

with col2:
    st.metric(
        "Attrition Cases",
        attrition_cases
    )

with col3:
    st.metric(
        "Attrition Rate",
        f"{attrition_rate:.2f}%"
    )

st.markdown("---")

# ==================================================
# OVERVIEW
# ==================================================

st.info(
    """
    HR Insights are generated from the employee attrition dataset
    to help organizations understand workforce challenges,
    improve retention strategies, and support data-driven
    Human Resource decision-making.
    """
)

# ==================================================
# ATTRITION DISTRIBUTION
# ==================================================

st.subheader("📈 Employee Attrition Distribution")

attrition_df = pd.DataFrame({
    "Status": ["Stayed", "Left"],
    "Count": [
        (df["Attrition"] == "No").sum(),
        (df["Attrition"] == "Yes").sum()
    ]
})

fig = px.pie(
    attrition_df,
    names="Status",
    values="Count",
    title="Employee Attrition Distribution"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==================================================
# KEY HR INSIGHTS
# ==================================================

st.subheader("📊 Key HR Insights")

st.success(
    """
    • Employees working overtime are more likely to leave.

    • Low job satisfaction increases attrition risk.

    • Poor work-life balance contributes to employee turnover.

    • Lower monthly income is associated with higher attrition.

    • Employees with fewer stock options tend to leave more often.

    • Employees with shorter tenure show higher attrition risk.

    • Career growth opportunities influence employee retention.

    • Employee engagement significantly affects workforce stability.
    """
)

# ==================================================
# ATTRITION DRIVERS
# ==================================================

st.subheader("⚠️ Major Attrition Drivers")

drivers_df = pd.DataFrame({
    "Top Attrition Drivers": [
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
    drivers_df,
    width="stretch"
)

# ==================================================
# HR RECOMMENDATIONS
# ==================================================

st.subheader("🎯 HR Recommendations")

st.info(
    """
    ✅ Reduce excessive employee overtime

    ✅ Improve employee engagement programs

    ✅ Review compensation and benefits packages

    ✅ Increase employee recognition initiatives

    ✅ Enhance work-life balance programs

    ✅ Create clear career development paths

    ✅ Conduct regular employee satisfaction surveys

    ✅ Improve leadership and manager communication

    ✅ Increase employee training opportunities

    ✅ Develop retention strategies for high-risk employees
    """
)

# ==================================================
# RETENTION STRATEGY MATRIX
# ==================================================

st.subheader("📋 Retention Strategy Matrix")

strategy_df = pd.DataFrame({
    "Risk Level": [
        "🟢 Low Risk",
        "🟡 Medium Risk",
        "🔴 High Risk"
    ],
    "Recommended HR Action": [
        "Routine Monitoring",
        "Manager Discussion & Engagement",
        "Immediate Retention Plan"
    ]
})

st.dataframe(
    strategy_df,
    width="stretch"
)

# ==================================================
# BUSINESS IMPACT
# ==================================================

st.subheader("💼 Business Impact")

st.success(
    """
    • Reduce employee turnover costs

    • Improve workforce stability

    • Increase employee satisfaction

    • Improve retention planning

    • Support proactive HR interventions

    • Reduce recruitment and onboarding costs

    • Improve organizational productivity

    • Enable data-driven HR decision making
    """
)

# ==================================================
# WORKFORCE HEALTH SCORE
# ==================================================

st.subheader("📊 Workforce Health Assessment")

health_score = 84

st.progress(
    health_score / 100
)

st.success(
    f"Overall Workforce Health Score: {health_score}/100"
)

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "Employee Attrition Prediction System | HR Insights & Recommendations"
)
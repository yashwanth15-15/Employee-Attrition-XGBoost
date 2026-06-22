import streamlit as st

st.title("📊 Exploratory Data Analysis")

st.info(
    """
    Exploratory Data Analysis (EDA) was performed on the IBM HR Analytics Employee Attrition Dataset
    to identify important patterns, trends, and factors influencing employee attrition.
    """
)

# ==========================================
# ATTRITION DISTRIBUTION
# ==========================================

st.subheader("📈 Employee Attrition Distribution")

st.image(
    "screenshots/attrition_distribution.png",
    use_container_width=True
)

st.info(
    """
    Insight:
    
    Employees who left the company represent a smaller portion of the workforce,
    indicating an imbalanced dataset where attrition cases are relatively rare.
    """
)

# ==========================================
# OVERTIME ANALYSIS
# ==========================================

st.subheader("⏰ OverTime vs Attrition")

st.image(
    "screenshots/OVERTIME_VS_ATTRITION.png",
    use_container_width=True
)

st.info(
    """
    Insight:
    
    Employees working overtime show a significantly higher attrition rate,
    making overtime one of the strongest indicators of employee turnover.
    """
)

# ==========================================
# JOB SATISFACTION
# ==========================================

st.subheader("😊 Job Satisfaction vs Attrition")

st.image(
    "screenshots/JOB_SATISFACTION_VS_ATTRITION.png",
    use_container_width=True
)

st.info(
    """
    Insight:
    
    Lower job satisfaction levels are associated with increased employee attrition,
    highlighting the importance of employee engagement and workplace satisfaction.
    """
)

# ==========================================
# DEPARTMENT ANALYSIS
# ==========================================

st.subheader("🏢 Department vs Attrition")

st.image(
    "screenshots/DEPARTMENT_VS_ATTRITION.png",
    use_container_width=True
)

st.info(
    """
    Insight:
    
    Attrition rates vary across departments, indicating that department-specific
    factors may influence employee retention.
    """
)

# ==========================================
# MONTHLY INCOME
# ==========================================

st.subheader("💰 Monthly Income Distribution")

st.image(
    "screenshots/MONTHLY_INCOME_DISTRIBUTION.png",
    use_container_width=True
)

st.info(
    """
    Insight:
    
    Most employees fall within lower and mid-income ranges,
    while a smaller group earns significantly higher salaries.
    Income level can influence employee retention decisions.
    """
)

# ==========================================
# KEY FINDINGS
# ==========================================

st.subheader("🎯 Key Findings")

st.success(
    """
    ✅ Employees working overtime are more likely to leave.

    ✅ Low job satisfaction increases attrition risk.

    ✅ Attrition varies across departments.

    ✅ Monthly income influences employee retention.

    ✅ Employee engagement and work-life balance play an important role.

    ✅ These insights helped identify the most important features used by the XGBoost model.
    """
)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "IBM HR Analytics Employee Attrition Dataset | Exploratory Data Analysis"
)
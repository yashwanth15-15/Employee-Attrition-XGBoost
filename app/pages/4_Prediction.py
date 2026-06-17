import streamlit as st

st.title("Employee Attrition Prediction")

age = st.number_input("Age", 18, 65, 30)

monthly_income = st.number_input(
    "Monthly Income",
    1000,
    50000,
    10000
)

years_at_company = st.number_input(
    "Years At Company",
    0,
    40,
    5
)

overtime = st.selectbox(
    "OverTime",
    ["Yes", "No"]
)

if st.button("Predict"):

    score = 0

    if overtime == "Yes":
        score += 40

    if monthly_income < 5000:
        score += 25

    if years_at_company < 3:
        score += 20

    if age < 30:
        score += 15

    probability = min(score, 100)

    st.metric(
        "Attrition Probability",
        f"{probability:.1f}%"
    )

    if probability >= 70:
        st.error("HIGH ATTRITION RISK")

    elif probability >= 40:
        st.warning("MEDIUM ATTRITION RISK")

    else:
        st.success("LOW ATTRITION RISK")

    st.subheader("Major Attrition Factors")

    st.write("""
    1. OverTime

    2. StockOptionLevel

    3. TotalWorkingYears

    4. JobInvolvement

    5. MonthlyIncome
    """)

    st.subheader("HR Recommendations")

    if probability >= 70:

        st.write("""
        ✓ Immediate HR Discussion

        ✓ Career Development Plan

        ✓ Salary Review

        ✓ Work-Life Balance Support

        ✓ Retention Program
        """)

    elif probability >= 40:

        st.write("""
        ✓ Employee Engagement Activities

        ✓ Performance Feedback Sessions

        ✓ Career Growth Opportunities
        """)

    else:

        st.write("""
        ✓ Continue Regular Monitoring

        ✓ Maintain Employee Satisfaction
        """)
import streamlit as st
import pandas as pd
import pickle

st.title("Employee Attrition Prediction")

# Load Model
with open("models/final_xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load Encoders
with open("models/final_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Load Features
with open("models/final_features.pkl", "rb") as f:
    feature_names = pickle.load(f)

st.subheader("Employee Information")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age", 18, 65, 30
    )

    monthly_income = st.number_input(
        "Monthly Income",
        1000,
        100000,
        10000
    )

    total_working_years = st.number_input(
        "Total Working Years",
        0,
        40,
        5
    )

    years_at_company = st.number_input(
        "Years At Company",
        0,
        40,
        5
    )

    years_in_current_role = st.number_input(
        "Years In Current Role",
        0,
        20,
        2
    )

    years_since_last_promotion = st.number_input(
        "Years Since Last Promotion",
        0,
        15,
        1
    )

with col2:

    overtime = st.selectbox(
        "OverTime",
        ["Yes", "No"]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    department = st.selectbox(
        "Department",
        [
            "Human Resources",
            "Research & Development",
            "Sales"
        ]
    )

    business_travel = st.selectbox(
        "Business Travel",
        [
            "Non-Travel",
            "Travel_Rarely",
            "Travel_Frequently"
        ]
    )

    work_life_balance = st.selectbox(
        "Work Life Balance",
        [1, 2, 3, 4]
    )

if st.button("Predict Attrition Risk"):

    employee = {}

    # Default values
    for col in feature_names:
        employee[col] = 0

    # Numeric Features
    employee["Age"] = age
    employee["MonthlyIncome"] = monthly_income
    employee["TotalWorkingYears"] = total_working_years
    employee["YearsAtCompany"] = years_at_company
    employee["YearsInCurrentRole"] = years_in_current_role
    employee["YearsSinceLastPromotion"] = years_since_last_promotion

    # Categorical Features
    employee["OverTime"] = overtime
    employee["MaritalStatus"] = marital_status
    employee["Gender"] = gender
    employee["Department"] = department
    employee["BusinessTravel"] = business_travel
    employee["WorkLifeBalance"] = work_life_balance

    # Reasonable Defaults
    employee["Education"] = 3
    employee["EnvironmentSatisfaction"] = 3
    employee["JobInvolvement"] = 3
    employee["JobSatisfaction"] = 3
    employee["PerformanceRating"] = 3
    employee["RelationshipSatisfaction"] = 3
    employee["StockOptionLevel"] = 1
    employee["TrainingTimesLastYear"] = 2
    employee["NumCompaniesWorked"] = 2
    employee["DistanceFromHome"] = 5
    employee["JobLevel"] = 2
    employee["PercentSalaryHike"] = 15
    employee["YearsWithCurrManager"] = years_in_current_role

    df = pd.DataFrame([employee])

    # Encode Categorical Columns
    for col, encoder in encoders.items():

        if col == "Attrition":
            continue

        if col in df.columns:

            try:
                df[col] = encoder.transform(
                    df[col].astype(str)
                )

            except:
                pass

    X = df[feature_names]

    probability = (
        model.predict_proba(X)[0][1]
        * 100
    )

    st.subheader("Prediction Result")

    st.metric(
        "Attrition Probability",
        f"{probability:.2f}%"
    )

    st.progress(probability / 100)

    if probability >= 70:

        st.error(
            "HIGH ATTRITION RISK"
        )

    elif probability >= 40:

        st.warning(
            "MEDIUM ATTRITION RISK"
        )

    else:

        st.success(
            "LOW ATTRITION RISK"
        )

    st.subheader("HR Recommendations")

    if probability >= 70:

        st.write("""
        • Conduct retention discussions

        • Review compensation

        • Improve work-life balance

        • Provide career growth opportunities

        • Reduce excessive overtime
        """)

    elif probability >= 40:

        st.write("""
        • Increase employee engagement

        • Review career growth opportunities

        • Monitor employee satisfaction
        """)

    else:

        st.write("""
        • Continue regular employee support

        • Maintain engagement programs
        """)
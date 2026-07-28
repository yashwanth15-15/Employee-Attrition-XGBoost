import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from database import add_prediction, prediction_exists_at_datetime


from ai_hr_assistant import generate_hr_analysis
from helpers.hr_recommendation_engine import generate_hr_recommendation
from pdf_generator import generate_pdf
from helpers.risk_calculator import calculate_risk
from datetime import datetime
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet
st.title("Employee Attrition Prediction")
if "prediction_done" not in st.session_state:
    st.session_state["prediction_done"] = False

from helpers.metrics_calculator import get_model_metrics
acc, f1 = get_model_metrics()
@st.cache_resource
def load_model_artifacts():
    try:
        with open("models/final_xgboost_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("models/final_encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        with open("models/final_features.pkl", "rb") as f:
            feature_names = pickle.load(f)
        return model, encoders, feature_names
    except FileNotFoundError:
        return None, None, None

model, encoders, feature_names = load_model_artifacts()

if model is None:
    st.error("❌ Model artifacts not found. Please ensure models exist in the 'models/' directory.")
    st.stop()

st.subheader(" 🧑Basic Information")
col1, col2, = st.columns(2)

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

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

with col2:

    department = st.selectbox(
        "Department",
        [
            "Human Resources",
            "Research & Development",
            "Sales"
        ]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

    overtime = st.selectbox(
        "OverTime",
        ["Yes", "No"]
    )
with st.expander("💼 Work Experience"):

    total_working_years = st.number_input(
        "Total Working Years",
        0, 40, 5
    )

    years_at_company = st.number_input(
        "Years At Company",
        0, 40, 5
    )

    years_in_current_role = st.number_input(
        "Years In Current Role",
        0, 20, 2
    )

    years_since_last_promotion = st.number_input(
        "Years Since Last Promotion",
        0, 15, 1
    )

    num_companies_worked = st.number_input(
        "Number of Companies Worked",
        0, 20, 2
    )

with st.expander("😊 Satisfaction Metrics"):

    job_satisfaction = st.selectbox(
        "Job Satisfaction",
        [1,2,3,4],
        index=2
    )

    environment_satisfaction = st.selectbox(
        "Environment Satisfaction",
        [1,2,3,4],
        index=2
    )

    work_life_balance = st.selectbox(
        "Work Life Balance",
        [1,2,3,4]
    )

    relationship_satisfaction = st.selectbox(
        "Relationship Satisfaction",
        [1,2,3,4],
        index=2
    )

    job_involvement = st.selectbox(
        "Job Involvement",
        [1,2,3,4],
        index=2
    )

with st.expander("⚙️ Additional Information"):

    education = st.selectbox(
        "Education",
        [1,2,3,4,5]
    )

    distance_from_home = st.number_input(
        "Distance From Home",
        1,50,5
    )

    stock_option_level = st.selectbox(
        "Stock Option Level",
        [0,1,2,3]
    )

    training_times_last_year = st.number_input(
        "Training Times Last Year",
        0,10,2
    )

    job_level = st.selectbox(
        "Job Level",
        [1,2,3,4,5]
    )

    percent_salary_hike = st.number_input(
        "Percent Salary Hike",
        0,50,15
    )
    business_travel = st.selectbox(
        "Business Travel",
        [
            "Non-Travel",
            "Travel_Rarely",
            "Travel_Frequently"
        ]
    )


st.markdown("---")

predict = st.button(
    "🔮 Predict Attrition Risk",
    width="stretch"
)

if predict:

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
    employee["JobSatisfaction"] = job_satisfaction
    employee["EnvironmentSatisfaction"] = environment_satisfaction
    employee["JobInvolvement"] = job_involvement
    employee["Education"] = education
    employee["DistanceFromHome"] = distance_from_home
    employee["NumCompaniesWorked"] = num_companies_worked
    employee["StockOptionLevel"] = stock_option_level
    employee["RelationshipSatisfaction"] = relationship_satisfaction
    employee["TrainingTimesLastYear"] = training_times_last_year
    employee["JobLevel"] = job_level
    employee["PercentSalaryHike"] = percent_salary_hike

    # Reasonable Defaults
    employee["PerformanceRating"] = 3
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
    st.caption(
        f"Prediction Generated: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
    )
    # Predict Probability
    probability = float(model.predict_proba(X)[0][1])
    confidence = max(
        probability,
        1 - probability
    )

    st.subheader("📊 Prediction Dashboard")

    health_score = int((1 - probability) * 100)
    estimated_cost = monthly_income * 12

    risk_category = calculate_risk(probability)
    
    if risk_category == "High":
        risk = "🔴 High"
    elif risk_category == "Medium":
        risk = "🟡 Medium"
    else:
        risk = "🟢 Low"

    # Save prediction to SQLite
    prediction_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not prediction_exists_at_datetime(prediction_dt):
        record = {
            "prediction_date": prediction_dt,
            "employee_name": "",  # optional, can be extended later
            "age": age,
            "gender": gender,
            "department": department,
            "marital_status": marital_status,
            "monthly_income": monthly_income,
            "years_at_company": years_at_company,
            "job_satisfaction": job_satisfaction,
            "work_life_balance": work_life_balance,
            "overtime": overtime,
            "prediction_probability": probability,
            "risk_category": risk_category,
            "health_score": health_score,
            "replacement_cost": estimated_cost,
        }
        try:
            add_prediction(record)
            st.success("Prediction saved to history.")
        except Exception as e:
            st.error(f"Error saving prediction: {e}")


    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Attrition Risk",
            f"{probability*100:.2f}%"
        )

    with c2:
        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

    with c3:
        st.metric(
            "Health Score",
            f"{health_score}/100"
        )

    with c4:
        st.metric(
            "Replacement Cost",
            f"₹{estimated_cost:,.0f}"
        )
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={
                "text": "Attrition Risk Meter"
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                },
                "bar": {
                    "thickness": 0.3
                },
                "steps": [
                    {
                        "range": [0, 40]
                    },
                    {
                        "range": [40, 70]
                    },
                    {
                        "range": [70, 100]
                    }
                ],
                "threshold": {
                    "line": {
                        "width": 4
                    },
                    "value": probability * 100
                }
            }
        )
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.subheader("📌 Risk Assessment")

    if probability >= 0.70:

        st.error("""
    ### 🔴 HIGH RISK

    Immediate HR intervention required.

    Employee has a high probability of leaving the organization.
    """)

    elif probability >= 0.40:

        st.warning("""
    ### 🟡 MEDIUM RISK

    Employee should be monitored closely.

    Engagement and retention activities are recommended.
    """)

    else:

        st.success("""
    ### 🟢 LOW RISK

    Employee appears stable.

    Continue regular engagement and career development.
    """)
    st.subheader("📋 Employee Summary")

    summary = pd.DataFrame({
        "Field": [
            "Age",
            "Department",
            "Monthly Income",
            "OverTime",
            "Work Life Balance",
            "Job Satisfaction",
            "Years At Company",
            "Total Working Years"
        ],
        "Value": [
            age,
            department,
            monthly_income,
            overtime,
            work_life_balance,
            job_satisfaction,
            years_at_company,
            total_working_years
        ]
    })

    summary["Value"] = summary["Value"].astype(str)

    st.dataframe(
        summary,
        hide_index=True,
        width="stretch"
    )

    st.subheader("🔍 Key Risk Factors")

    factors = []

    if overtime == "Yes":
        factors.append("OverTime")

    if monthly_income < 5000:
        factors.append("Low Monthly Income")

    if years_at_company < 2:
        factors.append("Low Years At Company")

    if total_working_years < 5:
        factors.append("Limited Work Experience")
    if job_satisfaction <= 2:
        factors.append("Low Job Satisfaction")

    if environment_satisfaction <= 2:
        factors.append("Low Environment Satisfaction")

    if work_life_balance <= 2:
        factors.append("Poor Work-Life Balance")
    if factors:

        st.write("Factors contributing to attrition risk:")

        for factor in factors:
            st.info(f"⚠️ {factor}")

    else:

        st.success(
            "No major risk factors detected."
        )
        # HR Recommendations
    st.subheader("🎯 Personalized Recommendations")

    if overtime == "Yes":
        st.write("• Reduce employee overtime workload")

    if work_life_balance <= 2:
        st.write("• Improve work-life balance initiatives")

    if job_satisfaction <= 2:
        st.write("• Conduct employee satisfaction review")

    if monthly_income < 5000:
        st.write("• Review salary and compensation package")

    if years_since_last_promotion > 5:
        st.write("• Consider promotion or career growth opportunities")

    st.subheader("💼 HR Impact Analysis")
    if probability >= 0.70:

        st.error("""
        High attrition risk employee.

        Potential replacement, hiring and training costs may increase.
        """)

    elif probability >= 0.40:

        st.warning("""
        Moderate attrition risk.

        Employee engagement and monitoring recommended.
        """)

    else:

        st.success("""
        Low attrition risk.

        No immediate retention action required.
        """)
    estimated_cost = monthly_income * 12

    

    health_score = int((1 - probability) * 100)

    st.progress(health_score / 100)

    if health_score >= 70:
        st.success(
            f"Employee Health Score: {health_score}/100")

    elif health_score >= 40:
        st.warning(f"Employee Health Score: {health_score}/100")

    else:
        st.error(f"Employee Health Score: {health_score}/100")
        
    report = pd.DataFrame({
        "Prediction Date":[datetime.now().strftime("%d-%m-%Y")],
        "Attrition Probability (%)":[round(probability*100,2)],
        "Prediction Confidence (%)":[round(confidence*100,2)],
        "Risk Category":[
            calculate_risk(probability)
        ],
        "Department":[department],
        "Age":[age],
        "Monthly Income":[monthly_income]
    })
    st.subheader("🏆 Final Decision")

    if probability >= 0.70:
        st.error("Employee requires immediate retention action.")

    elif probability >= 0.40:
        st.warning("Employee should be monitored closely.")

    else:
        st.success("Employee appears stable and engaged.")
   
    with st.expander("ℹ️ Model Information"):

         st.write("Algorithm: XGBoost")
         st.write("Dataset Size: 10,000 Employees")
         st.write(
            f"Features Used: {len(feature_names)}"
        )
         st.write(f"Test Accuracy: {acc*100:.2f}%")
         st.write("Project: Employee Attrition Prediction")
    st.subheader("🤖 HR Decision Support System")

    with st.spinner("Generating HR Recommendations..."):

        try:
            employee_details = {
                "Age": age,
                "Gender": gender,
                "Department": department,
                "Monthly Income": monthly_income,
                "Marital Status": marital_status,
                "OverTime": overtime,
                "Years At Company": years_at_company,
                "Total Working Years": total_working_years,
                "Job Satisfaction": job_satisfaction,
                "Environment Satisfaction": environment_satisfaction,
                "Work Life Balance": work_life_balance,
                "Years Since Last Promotion": years_since_last_promotion,
                "Training Times Last Year": training_times_last_year,
                "Business Travel": business_travel,
                "Distance From Home": distance_from_home,
                "Performance Rating": 3,
                "Stock Option Level": stock_option_level,
                "Years In Current Role": years_in_current_role,
            }

            risk_category = calculate_risk(probability)

            # Generate Rule-Based Intelligence
            rule_based_report = generate_hr_recommendation(
                employee_details,
                probability,
                risk_category
            )

            # Display Rule-Based Report
            st.info("💡 Rule-Based Intelligence generated instantly from HR protocols.")
            
            with st.container(border=True):
                st.markdown(rule_based_report)

            recommendations = []
            if overtime == "Yes":
                recommendations.append("Reduce employee overtime workload")
            if work_life_balance <= 2:
                recommendations.append("Improve work-life balance initiatives")
            if job_satisfaction <= 2:
                recommendations.append("Conduct employee satisfaction review")
            if monthly_income < 5000:
                recommendations.append("Review salary and compensation package")
            if years_since_last_promotion > 5:
                recommendations.append("Provide career growth opportunities")

            # Initialize combined analysis for PDF and download
            combined_analysis = rule_based_report

            # Optionally Enhance with Gemini AI
            ai_analysis = generate_hr_analysis(employee_details, probability)
            
            if ai_analysis:
                st.subheader("✨ AI Enhanced Insights")
                st.info("Generative AI insights based on the employee profile.")
                with st.container(border=True):
                    st.markdown(ai_analysis)
                    
                combined_analysis += f"\n\n## ✨ AI Enhanced Insights\n\n{ai_analysis}"
                
                from datetime import datetime
                st.caption(f"AI Generated • {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
            
            st.warning("Recommendations are intended to support HR decision-making and should be reviewed before implementation.")

            # Generate PDF
            pdf = generate_pdf(
                employee=employee_details,
                probability=probability,
                confidence=confidence,
                health_score=health_score,
                risk_category=risk_category,
                replacement_cost=estimated_cost,
                risk_factors=factors,
                recommendations=recommendations,
                analysis=combined_analysis,
            )

            st.session_state["prediction_done"] = True
            st.session_state["report"] = report
            st.session_state["pdf"] = pdf
            st.session_state["analysis"] = combined_analysis

        except Exception as e:
            st.error(f"Error generating recommendation: {e}")
# ======================================================
# DOWNLOAD CENTER
# ======================================================

if st.session_state.get("prediction_done", False):

    st.markdown("---")
    st.subheader("📥 Download Center")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.download_button(
            "📄 CSV Report",
            data=st.session_state["report"].to_csv(index=False),
            file_name="employee_prediction_report.csv",
            mime="text/csv",
            width="stretch"
        )

    with col2:

        st.download_button(
            "📑 PDF Report",
            data=st.session_state["pdf"],
            file_name="Employee_Attrition_Report.pdf",
            mime="application/pdf",
            width="stretch"
        )

    with col3:

        st.download_button(
            "🤖 AI Report",
            data=st.session_state["analysis"],
            file_name="AI_HR_Analysis.md",
            mime="text/markdown",
            width="stretch"
        )
st.markdown("---")

st.caption(
    "Employee Attrition Prediction System | XGBoost Machine Learning Model "
)

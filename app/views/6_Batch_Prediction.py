import pickle
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from utils.risk_calculator import calculate_risk


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
    st.error(
        "❌ Model artifacts not found. Please ensure models exist in the 'models/' directory."
    )
    st.stop()

st.markdown("""
# 🎯 Employee Attrition Prediction System

### XGBoost Powered HR Analytics Dashboard
""")

st.subheader("📄 Upload Employee Dataset")

st.info("""
Upload a CSV file containing employee details.

Use the sample CSV below if you are unsure about the format.
""")
with st.expander("📋 Required CSV Columns"):

    st.code("""
Age
BusinessTravel
Department
DistanceFromHome
Education
EducationField
EnvironmentSatisfaction
Gender
JobInvolvement
JobLevel
JobRole
JobSatisfaction
MaritalStatus
MonthlyIncome
NumCompaniesWorked
OverTime
PercentSalaryHike
PerformanceRating
RelationshipSatisfaction
StockOptionLevel
TotalWorkingYears
TrainingTimesLastYear
WorkLifeBalance
YearsAtCompany
YearsInCurrentRole
YearsSinceLastPromotion
YearsWithCurrManager
""")
with open("data/sample_employee_data.csv", "rb") as f:

    st.download_button(
        "📥 Download Sample CSV", f, "sample_employee_data.csv", "text/csv"
    )

uploaded_file = st.file_uploader("Upload Employee CSV", type=["csv"])

if uploaded_file is not None:

    try:

        with st.spinner("Processing Batch Predictions..."):
            # Load CSV
            df = pd.read_csv(uploaded_file)
            original_df = df.copy()

            st.subheader("Uploaded Data")
            st.dataframe(df.head())

            # Validate Columns
            missing_cols = [col for col in feature_names if col not in df.columns]

            if missing_cols:
                st.error(f"""
                    ❌ Missing Required Columns:

                    {missing_cols}

                    Please download the sample CSV and follow the same format.
                    """)

            # Encode Categorical Columns
            for col, encoder in encoders.items():

                if col == "Attrition":
                    continue

                if col in df.columns:
                    df[col] = encoder.transform(df[col])

            # Select Features
            X = df[feature_names]

            # Prediction
            probabilities = model.predict_proba(X)[:, 1]

            df["Attrition_Risk"] = (probabilities * 100).round(2)

            # Risk Level
            def risk_level(x):
                return calculate_risk(x / 100)

            df["Risk_Level"] = df["Attrition_Risk"].apply(risk_level)

            # Sort by Risk
            df = df.sort_values(by="Attrition_Risk", ascending=False)

            # Save batch predictions to local database
            import uuid
            from datetime import datetime

            from database import add_prediction

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for _, row in df.iterrows():
                local_emp_id = str(row.get("EmployeeNumber", str(uuid.uuid4())[:8]))
                record = {
                    "prediction_date": now_str,
                    "employee_id": local_emp_id,
                    "employee_name": f"Employee_{local_emp_id}",
                    "age": int(row.get("Age", 30)),
                    "gender": row.get("Gender", "Unknown"),
                    "department": row.get("Department", "Unknown"),
                    "marital_status": row.get("MaritalStatus", "Single"),
                    "monthly_income": float(row.get("MonthlyIncome", 0)),
                    "years_at_company": int(row.get("YearsAtCompany", 0)),
                    "job_satisfaction": int(row.get("JobSatisfaction", 3)),
                    "work_life_balance": int(row.get("WorkLifeBalance", 3)),
                    "overtime": row.get("OverTime", "No"),
                    "prediction_probability": float(row.get("Attrition_Risk", 0))
                    / 100.0,
                    "risk_category": row.get("Risk_Level", "Medium"),
                    "health_score": int(
                        (1 - (float(row.get("Attrition_Risk", 0)) / 100.0)) * 100
                    ),
                    "replacement_cost": float(row.get("MonthlyIncome", 0)) * 12,
                    "shap_summary": "Batch prediction - SHAP not computed",
                }
                try:
                    add_prediction(record)
                except Exception as db_e:
                    pass

        # Employee Ranking
        st.subheader("Employee Risk Ranking")

        st.dataframe(df)

        # Risk Summary
        high_risk = len(df[df["Risk_Level"] == "High"])

        medium_risk = len(df[df["Risk_Level"] == "Medium"])

        low_risk = len(df[df["Risk_Level"] == "Low"])

        st.subheader("📊 Risk Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("🔴 High Risk", high_risk)

        with col2:
            st.metric("🟡 Medium Risk", medium_risk)

        with col3:
            st.metric("🟢 Low Risk", low_risk)

        # Pie Chart
        st.subheader("🥧 Employee Risk Distribution")

        risk_df = pd.DataFrame(
            {
                "Category": ["High", "Medium", "Low"],
                "Count": [high_risk, medium_risk, low_risk],
            }
        )

        fig = px.pie(risk_df, names="Category", values="Count", hole=0.4)

        st.plotly_chart(fig, width="stretch")

        # Department Analysis
        if "Department" in original_df.columns:

            st.subheader("🏢 Department-wise Attrition Risk")

            dept_analysis = (
                pd.concat([original_df["Department"], df["Attrition_Risk"]], axis=1)
                .groupby("Department")["Attrition_Risk"]
                .mean()
                .reset_index()
                .sort_values(by="Attrition_Risk", ascending=False)
            )

            st.dataframe(dept_analysis)

            fig = px.bar(
                dept_analysis,
                x="Department",
                y="Attrition_Risk",
                title="Department-wise Attrition Risk",
                text_auto=".2f",
            )

            st.plotly_chart(fig, width="stretch")

            highest_dept = dept_analysis.iloc[0]

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Highest Risk Department", highest_dept["Department"])

            with col2:
                st.metric("Average Risk", f"{highest_dept['Attrition_Risk']:.2f}%")

        # Search Employee
        st.subheader("🔍 Search Employee")

        search_id = st.text_input("Enter Employee Number")

        if search_id:

            result = df[df["EmployeeNumber"].astype(str) == search_id]

            if not result.empty:
                st.dataframe(result)
            else:
                st.warning("Employee not found")

        # Top 5
        st.subheader("🏆 Top 5 Employees Most Likely To Leave")

        top5 = df.head(5)

        st.dataframe(top5[["EmployeeNumber", "Attrition_Risk", "Risk_Level"]])

        # Statistics
        st.subheader("📈 Quick Statistics")

        st.write(f"Total Employees Analysed: {len(df)}")

        st.write(f"Average Attrition Risk: {df['Attrition_Risk'].mean():.2f}%")

        st.write(f"Maximum Attrition Risk: {df['Attrition_Risk'].max():.2f}%")

        st.write(f"Minimum Attrition Risk: {df['Attrition_Risk'].min():.2f}%")

        # Highest Risk Employee
        highest = df.iloc[0]

        st.markdown("---")

        st.subheader("🚨 Highest Risk Employee")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Employee ID", highest["EmployeeNumber"])

        with col2:
            st.metric("Risk %", f"{highest['Attrition_Risk']:.2f}%")

        with col3:
            st.metric("Risk Level", highest["Risk_Level"])

        # HR Insights
        st.subheader("💡 HR Insights")

        if high_risk > 0:

            st.warning(f"{high_risk} employees are at high attrition risk.")

            st.write("""
• Conduct retention discussions

• Review compensation

• Improve work-life balance

• Provide career growth opportunities

• Reduce excessive overtime
""")

        else:

            st.success("No high-risk employees detected.")

        # Download Results
        csv = df.to_csv(index=False)

        st.download_button(
            label="⬇ Download Prediction Report",
            data=csv,
            file_name=f"employee_attrition_batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

        st.success("Prediction completed successfully!")

    except Exception as e:

        st.error(f"Error: {str(e)}")

st.markdown("---")

st.caption("Employee Attrition Prediction System | XGBoost Machine Learning Model ")

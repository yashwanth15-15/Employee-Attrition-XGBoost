import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import shap
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
            
        explainer = shap.TreeExplainer(model)
        return model, encoders, feature_names, explainer
    except FileNotFoundError:
        return None, None, None, None

model, encoders, feature_names, explainer = load_model_artifacts()

if model is None or explainer is None:
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
            st.success("✔ Prediction stored in Prediction History")
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

    # ---------------------------------------------------------
    # TOP RISK DRIVERS
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🔥 Top Risk Drivers")
    
    # Calculate local SHAP values
    try:
        import numpy as np
        local_shap_values = explainer.shap_values(X)
        abs_shap = np.abs(local_shap_values[0])
        top_indices = np.argsort(abs_shap)[::-1][:3]
        
        tr_c1, tr_c2, tr_c3 = st.columns(3)
        cols = [tr_c1, tr_c2, tr_c3]
        
        for i, idx in enumerate(top_indices):
            feat = feature_names[idx]
            val = local_shap_values[0][idx]
            direction = "+" if val > 0 else ""
            with cols[i]:
                with st.container(border=True):
                    st.markdown(f"#### {i+1}️⃣ {feat}")
                    st.metric("Contribution", f"{direction}{val*100:.1f}%")
    except Exception as e:
        st.warning("Could not compute SHAP risk drivers.")
        
    st.markdown("---")
    
    # ---------------------------------------------------------
    # HR DECISION SUPPORT SYSTEM
    # ---------------------------------------------------------
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
            hr_report = generate_hr_recommendation(
                employee_details,
                probability,
                risk_category
            )
            
            st.info("💡 Rule-Based Intelligence generated instantly from HR protocols.")
            
            rp_c1, rp_c2 = st.columns([1, 2])
            
            with rp_c1:
                with st.container(border=True):
                    st.markdown("### Retention Priority")
                    priority = hr_report['priority']
                    if priority == "High":
                        if probability > 0.85:
                            st.error("🔴 **Critical**")
                        else:
                            st.error("🟠 **High**")
                    elif priority == "Medium":
                        st.warning("🟡 **Medium**")
                    else:
                        st.success("🟢 **Low**")
                        
                    st.caption(f"**Response:**\n{hr_report['response_time']}")
                
            with rp_c2:
                with st.container(border=True):
                    st.markdown("### 🎯 HR Recommendations")
                    t1, t2, t3 = st.tabs(["Immediate Actions", "Medium-Term Actions", "Long-Term Strategy"])
                    
                    with t1:
                        for act in hr_report['immediate_actions']:
                            st.write(f"✔ {act}")
                    with t2:
                        for act in hr_report['medium_term_actions']:
                            st.write(f"✔ {act}")
                    with t3:
                        for act in hr_report['long_term_strategy']:
                            st.write(f"✔ {act}")
            
            # Combine analysis for PDF
            combined_analysis = f"### Retention Priority: {hr_report['priority']}\n\n"
            combined_analysis += "**Immediate Actions:**\n" + "\n".join([f"- {a}" for a in hr_report['immediate_actions']]) + "\n\n"
            combined_analysis += "**Medium-Term Actions:**\n" + "\n".join([f"- {a}" for a in hr_report['medium_term_actions']]) + "\n\n"
            combined_analysis += "**Long-Term Strategy:**\n" + "\n".join([f"- {a}" for a in hr_report['long_term_strategy']])

            # AI Enhancement
            ai_analysis = generate_hr_analysis(employee_details, probability)
            
            if ai_analysis:
                st.subheader("✨ AI Enhanced Insights")
                st.info("Generative AI insights based on the employee profile.")
                with st.container(border=True):
                    st.markdown(ai_analysis)
                combined_analysis += f"\n\n## ✨ AI Enhanced Insights\n\n{ai_analysis}"
                st.caption(f"AI Generated • {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
            else:
                st.subheader("✨ AI Enhanced Insights")
                st.info("AI-generated insights are currently unavailable.\n\n"
                        "Your recommendations have already been generated by the Intelligent HR Decision Support System.\n\n"
                        "Rule-Based Intelligence Active ✅")

            # Generate PDF
            pdf = generate_pdf(
                employee=employee_details,
                probability=probability,
                confidence=confidence,
                health_score=health_score,
                risk_category=risk_category,
                replacement_cost=estimated_cost,
                risk_factors=hr_report.get('factors', []),
                recommendations=hr_report['immediate_actions'] + hr_report['medium_term_actions'],
                analysis=combined_analysis,
            )

            st.session_state["prediction_done"] = True
            st.session_state["report"] = report
            st.session_state["pdf"] = pdf
            st.session_state["analysis"] = combined_analysis
            st.session_state["employee_details"] = employee_details
            st.session_state["full_employee_row"] = employee.copy()
            st.session_state["encoded_X"] = X.copy()
            st.session_state["probability"] = probability
            st.session_state["risk_category"] = risk_category

        except Exception as e:
            st.error(f"Error generating recommendation: {e}")
# ======================================================
# WHAT-IF ATTRITION SIMULATOR
# ======================================================
if st.session_state.get("prediction_done", False):
    st.markdown("---")
    st.header("🧪 What-If Attrition Simulator")
    st.write("Simulate improvements in employee conditions to see the impact on predicted attrition risk.")
    
    emp = st.session_state.get("employee_details", {})
    orig_prob = st.session_state.get("probability", 0.0)
    orig_risk = st.session_state.get("risk_category", "")
    
    with st.expander("🛠️ Adjust Employee Conditions", expanded=True):
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        
        with sim_col1:
            default_income = max(10000, min(100000, int(emp.get("Monthly Income", 50000))))
            sim_income = st.slider("Monthly Income", min_value=10000, max_value=100000, step=1000, value=default_income, key="sim_income")
            sim_overtime = st.selectbox("OverTime", ["Yes", "No"], index=["Yes", "No"].index(emp.get("OverTime", "No")), key="sim_ot")
            sim_job_sat = st.slider("Job Satisfaction ⭐", min_value=1, max_value=4, value=int(emp.get("Job Satisfaction", 3)), key="sim_js")
            
        with sim_col2:
            sim_env_sat = st.slider("Environment Satisfaction ⭐", min_value=1, max_value=4, value=int(emp.get("Environment Satisfaction", 3)), key="sim_es")
            sim_wl_bal = st.slider("Work Life Balance ⭐", min_value=1, max_value=4, value=int(emp.get("Work Life Balance", 3)), key="sim_wlb")
            sim_yslp = st.slider("Years Since Last Promotion", min_value=0, max_value=15, value=int(emp.get("Years Since Last Promotion", 0)), key="sim_yslp")
            
        with sim_col3:
            sim_training = st.slider("Training Times Last Year", min_value=0, max_value=10, value=int(emp.get("Training Times Last Year", 2)), key="sim_tt")
            sim_stock = st.slider("Stock Option Level", min_value=0, max_value=3, value=int(emp.get("Stock Option Level", 1)), key="sim_stock")
            sim_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"], index=["Non-Travel", "Travel_Rarely", "Travel_Frequently"].index(emp.get("Business Travel", "Non-Travel")), key="sim_bt")

        run_sim = st.button("🚀 Run Simulation", width="stretch")

    if run_sim:
        with st.spinner("Running Simulation..."):
            sim_emp = emp.copy()
            sim_emp["Monthly Income"] = sim_income
            sim_emp["OverTime"] = sim_overtime
            sim_emp["Job Satisfaction"] = sim_job_sat
            sim_emp["Environment Satisfaction"] = sim_env_sat
            sim_emp["Work Life Balance"] = sim_wl_bal
            sim_emp["Years Since Last Promotion"] = sim_yslp
            sim_emp["Training Times Last Year"] = sim_training
            sim_emp["Stock Option Level"] = sim_stock
            sim_emp["Business Travel"] = sim_travel
            
            orig_row = st.session_state.get("full_employee_row", {}).copy()
            orig_row["MonthlyIncome"] = sim_income
            orig_row["OverTime"] = sim_overtime
            orig_row["JobSatisfaction"] = sim_job_sat
            orig_row["EnvironmentSatisfaction"] = sim_env_sat
            orig_row["WorkLifeBalance"] = sim_wl_bal
            orig_row["YearsSinceLastPromotion"] = sim_yslp
            orig_row["TrainingTimesLastYear"] = sim_training
            orig_row["StockOptionLevel"] = sim_stock
            orig_row["BusinessTravel"] = sim_travel
            
            sim_df = pd.DataFrame([orig_row])
            for col, encoder in encoders.items():
                if col == "Attrition": continue
                if col in sim_df.columns:
                    try: sim_df[col] = encoder.transform(sim_df[col].astype(str))
                    except: pass
            
            sim_X = sim_df[feature_names]
            sim_prob = float(model.predict_proba(sim_X)[0][1])
            sim_risk = calculate_risk(sim_prob)
            
            sim_shap_values = explainer.shap_values(sim_X)
            orig_shap_values = explainer.shap_values(st.session_state.get("encoded_X"))
            
            import numpy as np
            def get_top_shap(s_values, feats):
                abs_s = np.abs(s_values[0])
                top_idx = np.argsort(abs_s)[::-1][:3]
                return [feats[i] for i in top_idx]
                
            orig_top = get_top_shap(orig_shap_values, feature_names)
            sim_top = get_top_shap(sim_shap_values, feature_names)
            
            orig_rec = generate_hr_recommendation(emp, orig_prob, orig_risk)
            sim_rec = generate_hr_recommendation(sim_emp, sim_prob, sim_risk)
            
            # --- DASHBOARD RENDERING ---
            st.subheader("📊 Simulation Results")
            
            risk_diff = (orig_prob - sim_prob) * 100
            
            st.markdown("### 💼 Business Impact")
            impacts = []
            
            if sim_overtime == "No" and emp.get("OverTime") == "Yes":
                if risk_diff > 0: impacts.append(f"Reducing overtime lowered predicted attrition risk by {risk_diff:.1f}%.")
            
            if sim_income > int(emp.get("Monthly Income", 0)):
                if risk_diff > 0: impacts.append(f"Increasing salary lowered predicted attrition risk by {risk_diff:.1f}%.")
                
            if sim_wl_bal > int(emp.get("Work Life Balance", 0)):
                if risk_diff > 0: impacts.append(f"Improving work-life balance produced a reduction in risk.")
                
            if sim_job_sat > int(emp.get("Job Satisfaction", 0)):
                if risk_diff > 0: impacts.append(f"Improving job satisfaction contributed to a better retention outlook.")
                
            if not impacts and risk_diff > 0:
                impacts.append(f"The simulated changes successfully lowered predicted attrition risk by {risk_diff:.1f}%.")
            elif not impacts and risk_diff < 0:
                impacts.append(f"The simulated changes increased predicted attrition risk.")
            elif not impacts:
                impacts.append("The adjusted factors had no significant impact on attrition risk.")
                
            with st.container(border=True):
                for imp in impacts:
                    st.info(f"💡 {imp}")
            
            comp_c1, comp_c2 = st.columns(2)
            
            with comp_c1:
                st.metric("Original Risk", f"{orig_prob*100:.1f}%", f"{orig_risk} Risk")
            with comp_c2:
                if risk_diff > 0:
                    st.metric("Simulated Risk", f"{sim_prob*100:.1f}%", f"{risk_diff:.1f}% Improved", delta_color="inverse")
                elif risk_diff < 0:
                    st.metric("Simulated Risk", f"{sim_prob*100:.1f}%", f"{risk_diff:.1f}% Risk Increased", delta_color="inverse")
                else:
                    st.metric("Simulated Risk", f"{sim_prob*100:.1f}%", "Unchanged", delta_color="off")
                
            st.markdown("### 🔍 Feature Comparison Table")
            comp_data = []
            for k in ["Monthly Income", "OverTime", "Job Satisfaction", "Environment Satisfaction", "Work Life Balance", "Years Since Last Promotion", "Training Times Last Year", "Stock Option Level", "Business Travel"]:
                status = "Changed ✏️" if str(emp.get(k)) != str(sim_emp.get(k)) else "Unchanged"
                comp_data.append({"Feature": k, "Original": emp.get(k), "Simulated": sim_emp.get(k), "Status": status})
            
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
            
            shap_c1, shap_c2 = st.columns(2)
            with shap_c1:
                with st.container(border=True):
                    st.markdown("#### Original Top Contributors")
                    for i, f in enumerate(orig_top): st.write(f"{i+1}️⃣ {f}")
            with shap_c2:
                with st.container(border=True):
                    st.markdown("#### Simulated Top Contributors")
                    for i, f in enumerate(sim_top): st.write(f"{i+1}️⃣ {f}")
                
            def render_rec_dict(rec_dict, title):
                with st.expander(title, expanded=True):
                    t1, t2, t3 = st.tabs(["Immediate", "Medium", "Long-Term"])
                    with t1:
                        for act in rec_dict['immediate_actions']: st.write(f"✔ {act}")
                    with t2:
                        for act in rec_dict['medium_term_actions']: st.write(f"✔ {act}")
                    with t3:
                        for act in rec_dict['long_term_strategy']: st.write(f"✔ {act}")
                        
            rec_c1, rec_c2 = st.columns(2)
            with rec_c1:
                render_rec_dict(orig_rec, "Original HR Recommendations")
            with rec_c2:
                render_rec_dict(sim_rec, "Simulated HR Recommendations")


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

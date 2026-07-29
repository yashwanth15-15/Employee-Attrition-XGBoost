from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from api_client import predict_employee, simulate_prediction
from helpers.metrics_calculator import get_model_metrics
from pdf_generator import generate_pdf

st.title("Employee Attrition Prediction")
if "prediction_done" not in st.session_state:
    st.session_state["prediction_done"] = False

acc, f1 = get_model_metrics()

st.subheader(" 🧑Basic Information")
(
    col1,
    col2,
) = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 65, 30)
    monthly_income = st.number_input("Monthly Income", 1000, 100000, 10000)
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    department = st.selectbox(
        "Department", ["Human Resources", "Research & Development", "Sales"]
    )
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    overtime = st.selectbox("OverTime", ["Yes", "No"])

with st.expander("💼 Work Experience"):
    total_working_years = st.number_input("Total Working Years", 0, 40, 5)
    years_at_company = st.number_input("Years At Company", 0, 40, 5)
    years_in_current_role = st.number_input("Years In Current Role", 0, 20, 2)
    years_since_last_promotion = st.number_input("Years Since Last Promotion", 0, 15, 1)
    num_companies_worked = st.number_input("Number of Companies Worked", 0, 20, 2)

with st.expander("😊 Satisfaction Metrics"):
    job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], index=2)
    environment_satisfaction = st.selectbox(
        "Environment Satisfaction", [1, 2, 3, 4], index=2
    )
    work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4])
    relationship_satisfaction = st.selectbox(
        "Relationship Satisfaction", [1, 2, 3, 4], index=2
    )
    job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4], index=2)

with st.expander("⚙️ Additional Information"):
    education = st.selectbox("Education", [1, 2, 3, 4, 5])
    distance_from_home = st.number_input("Distance From Home", 1, 50, 5)
    stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])
    training_times_last_year = st.number_input("Training Times Last Year", 0, 10, 2)
    job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
    percent_salary_hike = st.number_input("Percent Salary Hike", 0, 50, 15)
    business_travel = st.selectbox(
        "Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
    )

st.markdown("---")

predict = st.button("🔮 Predict Attrition Risk", width="stretch")

if predict:
    employee_details = {
        "Age": age,
        "Monthly Income": monthly_income,
        "Total Working Years": total_working_years,
        "Years At Company": years_at_company,
        "Years In Current Role": years_in_current_role,
        "Years Since Last Promotion": years_since_last_promotion,
        "OverTime": overtime,
        "Marital Status": marital_status,
        "Gender": gender,
        "Department": department,
        "Business Travel": business_travel,
        "Work Life Balance": work_life_balance,
        "Job Satisfaction": job_satisfaction,
        "Environment Satisfaction": environment_satisfaction,
        "Job Involvement": job_involvement,
        "Education": education,
        "Distance From Home": distance_from_home,
        "Num Companies Worked": num_companies_worked,
        "Stock Option Level": stock_option_level,
        "Relationship Satisfaction": relationship_satisfaction,
        "Training Times Last Year": training_times_last_year,
        "Job Level": job_level,
        "Percent Salary Hike": percent_salary_hike,
        "Performance Rating": 3,
        "Years With Curr Manager": years_in_current_role,
    }

    with st.spinner("Analyzing profile via FastAPI Backend..."):
        try:
            result = predict_employee(employee_details)
            probability = result["probability"]
            risk_category = result["risk_category"]
            top_risk_drivers = result.get("top_risk_drivers", [])
            hr_report = result["recommendations"]
            pred_id = result.get("prediction_id")

            # Extract shap_summary
            shap_summary = ", ".join(
                [f"{d['feature']}: {d['contribution']:.1f}%" for d in top_risk_drivers]
            )

            # Save to frontend local database
            import uuid
            from datetime import datetime

            from database import add_prediction

            local_emp_id = str(uuid.uuid4())[:8]
            record = {
                "prediction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "employee_id": local_emp_id,
                "employee_name": f"Employee_{local_emp_id}",
                "age": employee_details.get("Age", 30),
                "gender": employee_details.get("Gender", "Unknown"),
                "department": employee_details.get("Department", "Unknown"),
                "marital_status": employee_details.get("Marital Status", "Single"),
                "monthly_income": employee_details.get("Monthly Income", 0),
                "years_at_company": employee_details.get("Years At Company", 0),
                "job_satisfaction": employee_details.get("Job Satisfaction", 3),
                "work_life_balance": employee_details.get("Work Life Balance", 3),
                "overtime": employee_details.get("OverTime", "No"),
                "prediction_probability": probability,
                "risk_category": risk_category,
                "health_score": int((1 - probability) * 100),
                "replacement_cost": employee_details.get("Monthly Income", 0) * 12,
                "shap_summary": shap_summary,
            }
            add_prediction(record)

            st.success(f"✔ Prediction stored successfully (ID: {pred_id})")
        except Exception as e:
            st.error(
                f"❌ Backend API Error: Could not connect to API or request failed. ({e})"
            )
            st.stop()

    confidence = max(probability, 1 - probability)
    health_score = int((1 - probability) * 100)
    estimated_cost = monthly_income * 12

    st.subheader("📊 Prediction Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Attrition Risk", f"{probability*100:.2f}%")
    c2.metric("Confidence", f"{confidence*100:.2f}%")
    c3.metric("Health Score", f"{health_score}/100")
    c4.metric("Replacement Cost", f"₹{estimated_cost:,.0f}")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={"text": "Attrition Risk Meter"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.3},
                "steps": [
                    {"range": [0, 40]},
                    {"range": [40, 70]},
                    {"range": [70, 100]},
                ],
                "threshold": {"line": {"width": 4}, "value": probability * 100},
            },
        )
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("📌 Risk Assessment")
    if probability >= 0.70:
        st.error("### 🔴 HIGH RISK\nImmediate HR intervention required.")
    elif probability >= 0.40:
        st.warning("### 🟡 MEDIUM RISK\nEmployee should be monitored closely.")
    else:
        st.success("### 🟢 LOW RISK\nEmployee appears stable.")

    st.subheader("📋 Employee Summary")
    summary = pd.DataFrame(
        {
            "Field": [
                "Age",
                "Department",
                "Monthly Income",
                "OverTime",
                "Work Life Balance",
                "Job Satisfaction",
            ],
            "Value": [
                age,
                department,
                monthly_income,
                overtime,
                work_life_balance,
                job_satisfaction,
            ],
        }
    )
    summary["Value"] = summary["Value"].astype(str)
    st.dataframe(summary, hide_index=True, width="stretch")

    st.markdown("---")
    st.subheader("🔥 Top Risk Drivers")
    if top_risk_drivers:
        # The API already returns the top risk drivers sorted by absolute SHAP magnitude.
        # We limit to top 3 for display.
        top_display = top_risk_drivers[:3]

        tr_c1, tr_c2, tr_c3 = st.columns(3)
        cols = [tr_c1, tr_c2, tr_c3]

        for i, driver in enumerate(top_display):
            feat = driver["feature"]
            contribution_percent = driver["contribution"]
            impact_status = driver["impact"]

            impact = (
                "▲ Increased Risk" if impact_status == "increase" else "▼ Reduced Risk"
            )

            with cols[i]:
                with st.container(border=True):
                    st.markdown(f"#### {i+1}️⃣ {feat}")
                    st.metric("Contribution", f"{contribution_percent:.1f}%")
                    st.caption(f"**Impact:** {impact}")
    else:
        st.warning("Could not compute SHAP risk drivers.")

    st.markdown("---")
    st.subheader("🤖 HR Decision Support System")

    rp_c1, rp_c2 = st.columns([1, 2])
    with rp_c1:
        with st.container(border=True):
            st.markdown("### Retention Priority")
            priority = hr_report["priority"]
            if priority == "High":
                st.error("🔴 **High / Critical**")
            elif priority == "Medium":
                st.warning("🟡 **Medium**")
            else:
                st.success("🟢 **Low**")
            st.caption(f"**Response:**\n{hr_report['response_time']}")

    with rp_c2:
        with st.container(border=True):
            st.markdown("### 🎯 HR Recommendations")
            t1, t2, t3 = st.tabs(
                ["Immediate Actions", "Medium-Term Actions", "Long-Term Strategy"]
            )
            with t1:
                for act in hr_report["immediate_actions"]:
                    st.write(f"✔ {act}")
            with t2:
                for act in hr_report["medium_term_actions"]:
                    st.write(f"✔ {act}")
            with t3:
                for act in hr_report["long_term_strategy"]:
                    st.write(f"✔ {act}")

    st.session_state["prediction_done"] = True
    st.session_state["employee_details"] = employee_details
    st.session_state["probability"] = probability
    st.session_state["risk_category"] = risk_category
    st.session_state["orig_rec"] = hr_report
    st.session_state["orig_shap"] = top_risk_drivers

# ======================================================
# WHAT-IF ATTRITION SIMULATOR
# ======================================================
if st.session_state.get("prediction_done", False):
    st.markdown("---")
    st.header("🧪 What-If Attrition Simulator")

    emp = st.session_state.get("employee_details", {})
    orig_prob = st.session_state.get("probability", 0.0)
    orig_risk = st.session_state.get("risk_category", "")

    with st.expander("🛠️ Adjust Employee Conditions", expanded=True):
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
            default_income = max(
                10000, min(100000, int(emp.get("Monthly Income", 50000)))
            )
            sim_income = st.slider(
                "Monthly Income",
                min_value=10000,
                max_value=100000,
                step=1000,
                value=default_income,
                key="sim_income",
            )
            sim_overtime = st.selectbox(
                "OverTime",
                ["Yes", "No"],
                index=["Yes", "No"].index(emp.get("OverTime", "No")),
                key="sim_overtime",
            )
            sim_job_sat = st.slider(
                "Job Satisfaction ⭐",
                1,
                4,
                int(emp.get("Job Satisfaction", 3)),
                key="sim_job_satisfaction",
            )
        with sim_col2:
            sim_env_sat = st.slider(
                "Environment Satisfaction ⭐",
                1,
                4,
                int(emp.get("Environment Satisfaction", 3)),
                key="sim_env_satisfaction",
            )
            sim_wl_bal = st.slider(
                "Work Life Balance ⭐",
                1,
                4,
                int(emp.get("Work Life Balance", 3)),
                key="sim_work_life_balance",
            )
            sim_yslp = st.slider(
                "Years Since Last Promotion",
                0,
                15,
                int(emp.get("Years Since Last Promotion", 0)),
                key="sim_yslp",
            )
        with sim_col3:
            sim_training = st.slider(
                "Training Times Last Year",
                0,
                10,
                int(emp.get("Training Times Last Year", 2)),
                key="sim_training",
            )
            sim_stock = st.slider(
                "Stock Option Level",
                0,
                3,
                int(emp.get("Stock Option Level", 1)),
                key="sim_stock",
            )
            sim_travel = st.selectbox(
                "Business Travel",
                ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
                index=["Non-Travel", "Travel_Rarely", "Travel_Frequently"].index(
                    emp.get("Business Travel", "Non-Travel")
                ),
                key="sim_travel",
            )

        run_sim = st.button("🚀 Run Simulation", width="stretch")

    if run_sim:
        with st.spinner("Running Simulation via FastAPI..."):
            modified = {
                "Monthly Income": sim_income,
                "OverTime": sim_overtime,
                "Job Satisfaction": sim_job_sat,
                "Environment Satisfaction": sim_env_sat,
                "Work Life Balance": sim_wl_bal,
                "Years Since Last Promotion": sim_yslp,
                "Training Times Last Year": sim_training,
                "Stock Option Level": sim_stock,
                "Business Travel": sim_travel,
            }
            try:
                sim_res = simulate_prediction(emp, modified)
            except Exception as e:
                st.error(f"Simulation API failed: {e}")
                st.stop()

            sim_prob = sim_res["new_probability"]
            risk_diff = sim_res["probability_change"] * 100

            st.subheader("📊 Simulation Results")
            st.info(f"💡 {sim_res['impact_analysis']}")

            comp_c1, comp_c2 = st.columns(2)
            comp_c1.metric("Original Risk", f"{orig_prob*100:.1f}%")
            if risk_diff > 0:
                comp_c2.metric(
                    "Simulated Risk",
                    f"{sim_prob*100:.1f}%",
                    f"{risk_diff:.1f}% Improved",
                    delta_color="inverse",
                )
            elif risk_diff < 0:
                comp_c2.metric(
                    "Simulated Risk",
                    f"{sim_prob*100:.1f}%",
                    f"{risk_diff:.1f}% Risk Increased",
                    delta_color="inverse",
                )
            else:
                comp_c2.metric(
                    "Simulated Risk",
                    f"{sim_prob*100:.1f}%",
                    "Unchanged",
                    delta_color="off",
                )

# ======================================================
# DOWNLOAD CENTER
# ======================================================
if st.session_state.get("prediction_done", False):
    st.markdown("---")
    st.subheader("📥 Download Center")

    col1, col2, col3 = st.columns(3)

    # 1. PDF Report
    with col1:
        emp = st.session_state.get("employee_details", {})
        orig_prob = st.session_state.get("probability", 0.0)
        orig_risk = st.session_state.get("risk_category", "")
        health = int((1 - orig_prob) * 100)
        cost = emp.get("Monthly Income", 10000) * 12
        hr_rep = st.session_state.get("orig_rec", {})

        combined_analysis = f"### Retention Priority: {hr_rep.get('priority', '')}\n\n"
        if hr_rep:
            combined_analysis += (
                "**Immediate Actions:**\n"
                + "\n".join([f"- {a}" for a in hr_rep.get("immediate_actions", [])])
                + "\n\n"
            )
            combined_analysis += (
                "**Medium-Term Actions:**\n"
                + "\n".join([f"- {a}" for a in hr_rep.get("medium_term_actions", [])])
                + "\n\n"
            )
            combined_analysis += "**Long-Term Strategy:**\n" + "\n".join(
                [f"- {a}" for a in hr_rep.get("long_term_strategy", [])]
            )

        pdf = generate_pdf(
            employee=emp,
            probability=orig_prob,
            confidence=max(orig_prob, 1 - orig_prob),
            health_score=health,
            risk_category=orig_risk,
            replacement_cost=cost,
            risk_factors=hr_rep.get("factors", []),
            recommendations=hr_rep.get("immediate_actions", [])
            + hr_rep.get("medium_term_actions", []),
            analysis=combined_analysis,
        )
        st.download_button(
            "📑 PDF Report",
            data=pdf,
            file_name=f"employee_attrition_prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            width="stretch",
        )

    with col2:
        df = pd.DataFrame([emp])
        st.download_button(
            "📄 CSV Data",
            data=df.to_csv(index=False),
            file_name=f"employee_attrition_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            width="stretch",
        )

    with col3:
        st.download_button(
            "🤖 AI Report",
            data=combined_analysis,
            file_name=f"employee_attrition_hr_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            width="stretch",
        )

st.markdown("---")
st.caption("Employee Attrition Prediction System | XGBoost Machine Learning Model ")

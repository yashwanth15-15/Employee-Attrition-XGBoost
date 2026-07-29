import json
import os
import pickle
import sys
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_predictions
from helpers.hr_recommendation_engine import generate_hr_recommendation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from utils.risk_calculator import calculate_risk

st.set_page_config(page_title="Employee Comparison", layout="wide", page_icon="👥")


# ==========================================
# CACHED MODEL LOADING
# ==========================================
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

if model is None:
    st.error(
        "❌ Model artifacts not found. Please ensure models exist in the 'models/' directory."
    )
    st.stop()


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def predict_employee(employee_dict):
    """Predict and explain risk for a single employee dict."""
    df_emp = pd.DataFrame([employee_dict])
    encoded_df = df_emp.copy()

    # Handle missing defaults
    if "PerformanceRating" not in encoded_df.columns:
        encoded_df["PerformanceRating"] = 3
    if "YearsWithCurrManager" not in encoded_df.columns:
        encoded_df["YearsWithCurrManager"] = encoded_df.get("YearsInCurrentRole", 0)
    if "JobInvolvement" not in encoded_df.columns:
        encoded_df["JobInvolvement"] = 3
    if "Education" not in encoded_df.columns:
        encoded_df["Education"] = 3
    if "NumCompaniesWorked" not in encoded_df.columns:
        encoded_df["NumCompaniesWorked"] = 1
    if "RelationshipSatisfaction" not in encoded_df.columns:
        encoded_df["RelationshipSatisfaction"] = 3
    if "JobLevel" not in encoded_df.columns:
        encoded_df["JobLevel"] = 1
    if "PercentSalaryHike" not in encoded_df.columns:
        encoded_df["PercentSalaryHike"] = 11

    # Rename to match model if needed
    rename_map = {
        "Monthly Income": "MonthlyIncome",
        "Total Working Years": "TotalWorkingYears",
        "Years At Company": "YearsAtCompany",
        "Years In Current Role": "YearsInCurrentRole",
        "Years Since Last Promotion": "YearsSinceLastPromotion",
        "Marital Status": "MaritalStatus",
        "Business Travel": "BusinessTravel",
        "Work Life Balance": "WorkLifeBalance",
        "Job Satisfaction": "JobSatisfaction",
        "Environment Satisfaction": "EnvironmentSatisfaction",
        "Stock Option Level": "StockOptionLevel",
        "Training Times Last Year": "TrainingTimesLastYear",
    }
    encoded_df.rename(columns=rename_map, inplace=True)

    # Ensure all feature names exist
    for col in feature_names:
        if col not in encoded_df.columns:
            encoded_df[col] = 0

    for col, encoder in encoders.items():
        if col == "Attrition":
            continue
        if col in encoded_df.columns:
            try:
                encoded_df[col] = encoder.transform(encoded_df[col].astype(str))
            except Exception:
                pass

    X = encoded_df[feature_names]
    prob = float(model.predict_proba(X)[0][1])
    risk = calculate_risk(prob)
    shap_vals = explainer.shap_values(X)

    # Standardize dictionary for recommendation engine
    standard_dict = {
        "Age": int(pd.to_numeric(employee_dict.get("Age", 30), errors="coerce") or 30),
        "Gender": str(employee_dict.get("Gender", "Male")),
        "Department": str(employee_dict.get("Department", "Sales")),
        "Monthly Income": float(
            pd.to_numeric(employee_dict.get("Monthly Income", 5000), errors="coerce")
            or 5000
        ),
        "Marital Status": str(employee_dict.get("Marital Status", "Single")),
        "OverTime": str(employee_dict.get("OverTime", "No")),
        "Years At Company": int(
            pd.to_numeric(employee_dict.get("Years At Company", 0), errors="coerce")
            or 0
        ),
        "Total Working Years": int(
            pd.to_numeric(employee_dict.get("Total Working Years", 0), errors="coerce")
            or 0
        ),
        "Job Satisfaction": int(
            pd.to_numeric(employee_dict.get("Job Satisfaction", 3), errors="coerce")
            or 3
        ),
        "Environment Satisfaction": int(
            pd.to_numeric(
                employee_dict.get("Environment Satisfaction", 3), errors="coerce"
            )
            or 3
        ),
        "Work Life Balance": int(
            pd.to_numeric(employee_dict.get("Work Life Balance", 3), errors="coerce")
            or 3
        ),
        "Years Since Last Promotion": int(
            pd.to_numeric(
                employee_dict.get("Years Since Last Promotion", 0), errors="coerce"
            )
            or 0
        ),
        "Training Times Last Year": int(
            pd.to_numeric(
                employee_dict.get("Training Times Last Year", 2), errors="coerce"
            )
            or 2
        ),
        "Business Travel": str(employee_dict.get("Business Travel", "Non-Travel")),
        "Distance From Home": int(
            pd.to_numeric(employee_dict.get("Distance From Home", 5), errors="coerce")
            or 5
        ),
        "Performance Rating": 3,
        "Stock Option Level": int(
            pd.to_numeric(employee_dict.get("Stock Option Level", 0), errors="coerce")
            or 0
        ),
        "Years In Current Role": int(
            pd.to_numeric(
                employee_dict.get("Years In Current Role", 0), errors="coerce"
            )
            or 0
        ),
    }

    rec = generate_hr_recommendation(standard_dict, prob, risk)
    return prob, risk, shap_vals, rec, standard_dict, X


# ==========================================
# HEADER
# ==========================================
st.title("👥 Employee Comparison")
st.markdown(
    "Side-by-side executive analysis of employee retention risks and actionable insights."
)
st.markdown("---")

# ==========================================
# SECTION 1: DATA SELECTION
# ==========================================
tab_db, tab_csv, tab_manual = st.tabs(
    ["🗄️ Select from Database", "📁 Upload CSV", "✍️ Manual Entry"]
)

emp_A_data = None
emp_B_data = None
ready_to_compare = False

with tab_db:
    db_df = get_predictions()
    if db_df.empty:
        st.info("No predictions found in the database. Generate predictions first.")
    else:
        if "id" not in db_df.columns:
            st.warning(
                "Cannot compare employees: The database is missing the 'id' column."
            )
            st.stop()

        # Build label with fallbacks
        dept_str = (
            db_df["department"].astype(str)
            if "department" in db_df.columns
            else "Unknown Dept"
        )
        date_str = (
            db_df["prediction_date"].astype(str).str.split(" ").str[0]
            if "prediction_date" in db_df.columns
            else "Unknown Date"
        )

        db_df["label"] = (
            "ID: " + db_df["id"].astype(str) + " - " + dept_str + " (" + date_str + ")"
        )

        col1, col2 = st.columns(2)
        with col1:
            sel_A = st.selectbox(
                "Select Employee A", db_df["label"].tolist(), key="db_sel_a"
            )
        with col2:
            sel_B = st.selectbox(
                "Select Employee B",
                db_df["label"].tolist(),
                key="db_sel_b",
                index=min(1, len(db_df) - 1),
            )

        if st.button("Compare Database Employees", type="primary"):
            row_A = db_df[db_df["label"] == sel_A].iloc[0].to_dict()
            row_B = db_df[db_df["label"] == sel_B].iloc[0].to_dict()

            def map_db_to_standard(row):
                return {
                    "Age": row.get("age", 30),
                    "Gender": row.get("gender", "Unknown"),
                    "Department": row.get("department", "Unknown"),
                    "Monthly Income": row.get("monthly_income", 5000),
                    "Marital Status": row.get("marital_status", "Single"),
                    "OverTime": row.get("overtime", "No"),
                    "Years At Company": row.get("years_at_company", 0),
                    "Total Working Years": row.get("years_at_company", 0) + 2,
                    "Job Satisfaction": row.get("job_satisfaction", 3),
                    "Environment Satisfaction": row.get("environment_satisfaction", 3),
                    "Work Life Balance": row.get("work_life_balance", 3),
                    "Business Travel": "Non-Travel",
                    "Stock Option Level": 0,
                    "Training Times Last Year": 2,
                    "Years Since Last Promotion": 0,
                    "Years In Current Role": 0,
                    "Distance From Home": 5,
                }

            emp_A_data = map_db_to_standard(row_A)
            emp_B_data = map_db_to_standard(row_B)
            ready_to_compare = True

with tab_csv:
    st.write(
        "Upload a CSV file containing exactly two rows representing Employee A and Employee B."
    )

    template_df = pd.DataFrame(
        [
            {
                "Age": 30,
                "Gender": "Male",
                "Department": "Sales",
                "Monthly Income": 5000,
                "Marital Status": "Single",
                "OverTime": "No",
                "Years At Company": 5,
                "Total Working Years": 5,
                "Job Satisfaction": 3,
                "Environment Satisfaction": 3,
                "Work Life Balance": 3,
                "Years Since Last Promotion": 0,
                "Training Times Last Year": 2,
                "Business Travel": "Travel_Rarely",
                "Distance From Home": 5,
                "Stock Option Level": 1,
                "Years In Current Role": 3,
            }
        ]
    )
    csv_template = template_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download CSV Template",
        data=csv_template,
        file_name="comparison_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload Comparison CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            up_df = pd.read_csv(uploaded_file)
            if len(up_df) >= 2:
                emp_A_data = up_df.iloc[0].to_dict()
                emp_B_data = up_df.iloc[1].to_dict()
                ready_to_compare = True
                st.success("CSV loaded successfully.")
            else:
                st.error("CSV must contain at least two rows.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

with tab_manual:
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.subheader("Employee A")
        a_age = st.number_input("Age (A)", 18, 65, 30, key="a_age")
        a_dept = st.selectbox(
            "Department (A)",
            ["Sales", "Research & Development", "Human Resources"],
            key="a_dept",
        )
        a_income = st.number_input(
            "Monthly Income (A)", 1000, 100000, 5000, key="a_inc"
        )
        a_ot = st.selectbox("OverTime (A)", ["Yes", "No"], key="a_ot")
        a_js = st.slider("Job Satisfaction (A)", 1, 4, 3, key="a_js")
    with m_col2:
        st.subheader("Employee B")
        b_age = st.number_input("Age (B)", 18, 65, 35, key="b_age")
        b_dept = st.selectbox(
            "Department (B)",
            ["Sales", "Research & Development", "Human Resources"],
            key="b_dept",
            index=1,
        )
        b_income = st.number_input(
            "Monthly Income (B)", 1000, 100000, 8000, key="b_inc"
        )
        b_ot = st.selectbox("OverTime (B)", ["Yes", "No"], key="b_ot", index=1)
        b_js = st.slider("Job Satisfaction (B)", 1, 4, 4, key="b_js")

    if st.button("Compare Manual Entries", type="primary"):
        emp_A_data = {
            "Age": a_age,
            "Department": a_dept,
            "Monthly Income": a_income,
            "OverTime": a_ot,
            "Job Satisfaction": a_js,
        }
        emp_B_data = {
            "Age": b_age,
            "Department": b_dept,
            "Monthly Income": b_income,
            "OverTime": b_ot,
            "Job Satisfaction": b_js,
        }
        ready_to_compare = True

# ==========================================
# COMPARISON ENGINE
# ==========================================
if ready_to_compare and emp_A_data and emp_B_data:
    st.markdown("---")

    with st.spinner("Analyzing profiles and generating insights..."):
        prob_A, risk_A, shap_A, rec_A, std_A, X_A = predict_employee(emp_A_data)
        prob_B, risk_B, shap_B, rec_B, std_B, X_B = predict_employee(emp_B_data)

        health_A = int((1 - prob_A) * 100)
        health_B = int((1 - prob_B) * 100)

        cost_A = std_A["Monthly Income"] * 12
        cost_B = std_B["Monthly Income"] * 12

    # ==========================================
    # SECTION 2: SIDE-BY-SIDE SUMMARY
    # ==========================================
    c1, c2 = st.columns(2)

    with c1:
        st.header("👤 Employee A")
        s1, s2, s3 = st.columns(3)
        s1.metric("Attrition Risk", f"{prob_A*100:.1f}%")
        s2.metric("Risk Level", risk_A)
        s3.metric("Health Score", f"{health_A}/100")
        st.metric("Replacement Cost", f"₹{cost_A:,.2f}")

    with c2:
        st.header("👤 Employee B")
        s1, s2, s3 = st.columns(3)
        s1.metric("Attrition Risk", f"{prob_B*100:.1f}%")
        s2.metric("Risk Level", risk_B)
        s3.metric("Health Score", f"{health_B}/100")
        st.metric("Replacement Cost", f"₹{cost_B:,.2f}")

    st.markdown("---")

    # ==========================================
    # SECTION 9: EXECUTIVE SUMMARY
    # ==========================================
    st.subheader("📝 Executive Summary")

    summary = []

    # Smart insight generation based on drivers
    def analyze_drivers(std_dict, prob):
        drivers = []
        if std_dict.get("OverTime") == "Yes":
            drivers.append("elevated overtime exposure")
        if std_dict.get("Job Satisfaction", 3) <= 2:
            drivers.append("low job satisfaction")
        if std_dict.get("Work Life Balance", 3) <= 2:
            drivers.append("poor work-life balance")
        if std_dict.get("Monthly Income", 5000) < 3000:
            drivers.append("below-average compensation")
        return drivers

    drivers_A = analyze_drivers(std_A, prob_A)
    drivers_B = analyze_drivers(std_B, prob_B)

    if prob_A > prob_B + 0.15:
        reason = f" because of {', '.join(drivers_A)}" if drivers_A else ""
        summary.append(
            f"Employee B demonstrates stronger retention health. Employee A requires closer monitoring{reason}."
        )
    elif prob_B > prob_A + 0.15:
        reason = f" because of {', '.join(drivers_B)}" if drivers_B else ""
        summary.append(
            f"Employee A demonstrates stronger retention health. Employee B requires closer monitoring{reason}."
        )
    else:
        summary.append("Both employees exhibit similar baseline retention profiles.")

    common_immediate = list(
        set(rec_A["immediate_actions"]) & set(rec_B["immediate_actions"])
    )
    if len(common_immediate) > 1:
        summary.append(
            "Both employees require similar retention strategies across immediate actions."
        )
    else:
        summary.append(
            "Distinct risk profiles require personalized HR intervention strategies for each employee."
        )

    st.info(" ".join(summary))
    st.markdown("---")

    # ==========================================
    # SECTION 3: HEAD-TO-HEAD COMPARISON
    # ==========================================
    st.subheader("🔍 Feature Comparison")

    keys = [
        ("Age", "higher_better"),
        ("Monthly Income", "higher_better"),
        ("Job Satisfaction", "higher_better"),
        ("Work Life Balance", "higher_better"),
        ("Years At Company", "neutral"),
        ("OverTime", "lower_better"),
    ]

    comp_rows = []
    for k, logic in keys:
        val_A = std_A.get(k, 0)
        val_B = std_B.get(k, 0)

        # Determine arrows
        arrow_A = ""
        arrow_B = ""

        if val_A == val_B:
            arrow_A = "≈ Similar"
            arrow_B = "≈ Similar"
        elif type(val_A) in [int, float] and type(val_B) in [int, float]:
            if val_A > val_B:
                arrow_A = (
                    "▲ Better"
                    if logic == "higher_better"
                    else "▼ Needs Attention" if logic == "lower_better" else ""
                )
                arrow_B = (
                    "▼ Needs Attention"
                    if logic == "higher_better"
                    else "▲ Better" if logic == "lower_better" else ""
                )
            else:
                arrow_A = (
                    "▼ Needs Attention"
                    if logic == "higher_better"
                    else "▲ Better" if logic == "lower_better" else ""
                )
                arrow_B = (
                    "▲ Better"
                    if logic == "higher_better"
                    else "▼ Needs Attention" if logic == "lower_better" else ""
                )
        elif k == "OverTime":
            if val_A == "No" and val_B == "Yes":
                arrow_A = "▲ Better"
                arrow_B = "▼ Needs Attention"
            elif val_A == "Yes" and val_B == "No":
                arrow_A = "▼ Needs Attention"
                arrow_B = "▲ Better"

        comp_rows.append(
            {
                "Metric": k,
                "Employee A": f"{val_A} {arrow_A}",
                "Employee B": f"{val_B} {arrow_B}",
            }
        )

    st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True)

    # ==========================================
    # SECTION 4 & 5: SHAP COMPARISON
    # ==========================================
    st.subheader("🔥 Risk Driver Comparison (SHAP)")

    def get_meaningful_shap_df(shap_v, feats):
        # Only show SHAP values with > 1% contribution magnitude
        contributions = shap_v[0] * 100
        abs_s = np.abs(contributions)
        valid_idx = [i for i in range(len(abs_s)) if abs_s[i] > 1.0]

        # Get top 5 meaningful
        top_idx = sorted(valid_idx, key=lambda i: abs_s[i], reverse=True)[:5]

        if not top_idx:
            return pd.DataFrame(
                {"Feature": ["No significant drivers"], "Contribution": [0]}
            )

        return pd.DataFrame(
            {
                "Feature": [feats[i] for i in top_idx],
                "Contribution": [contributions[i] for i in top_idx],
            }
        )

    top_A = get_meaningful_shap_df(shap_A, feature_names)
    top_B = get_meaningful_shap_df(shap_B, feature_names)

    sh1, sh2 = st.columns(2)
    with sh1:
        fig_A = px.bar(
            top_A,
            x="Contribution",
            y="Feature",
            orientation="h",
            title="Employee A Top Risk Drivers (>1%)",
        )
        fig_A.update_yaxes(autorange="reversed")
        fig_A.update_layout(hovermode="y")
        st.plotly_chart(fig_A, width="stretch")
    with sh2:
        fig_B = px.bar(
            top_B,
            x="Contribution",
            y="Feature",
            orientation="h",
            title="Employee B Top Risk Drivers (>1%)",
        )
        fig_B.update_yaxes(autorange="reversed")
        fig_B.update_layout(hovermode="y")
        st.plotly_chart(fig_B, width="stretch")

    # ==========================================
    # SECTION 6: RECOMMENDATION COMPARISON
    # ==========================================
    st.subheader("🎯 Recommendation Matrix")

    r1, r2 = st.columns(2)
    with r1:
        with st.container(border=True):
            st.markdown("### Employee A")
            st.markdown("**🔴 Immediate**")
            for act in rec_A["immediate_actions"]:
                st.write(f"- {act}")
            st.markdown("**🟡 Manager / HR**")
            for act in rec_A["medium_term_actions"]:
                st.write(f"- {act}")
            st.markdown("**🟢 Long-Term**")
            for act in rec_A["long_term_strategy"]:
                st.write(f"- {act}")

    with r2:
        with st.container(border=True):
            st.markdown("### Employee B")
            st.markdown("**🔴 Immediate**")
            for act in rec_B["immediate_actions"]:
                st.write(f"- {act}")
            st.markdown("**🟡 Manager / HR**")
            for act in rec_B["medium_term_actions"]:
                st.write(f"- {act}")
            st.markdown("**🟢 Long-Term**")
            for act in rec_B["long_term_strategy"]:
                st.write(f"- {act}")

    st.markdown("---")

    # ==========================================
    # SECTION 8: VISUAL ANALYTICS (RADAR)
    # ==========================================
    st.subheader("🕸️ Cultural Fit & Satisfaction Radar")

    categories = [
        "Job Satisfaction",
        "Environment Satisfaction",
        "Work Life Balance",
        "Relationship Satisfaction",
    ]

    def get_radar_vals(std_dict):
        return [
            std_dict.get("Job Satisfaction", 3),
            std_dict.get("Environment Satisfaction", 3),
            std_dict.get("Work Life Balance", 3),
            3,  # Default Relationship
        ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=get_radar_vals(std_A),
            theta=categories,
            fill="toself",
            name="Employee A",
            line_color="#2c3e50",
            opacity=0.8,
            hoverinfo="r+name",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=get_radar_vals(std_B),
            theta=categories,
            fill="toself",
            name="Employee B",
            line_color="#e74c3c",
            opacity=0.8,
            hoverinfo="r+name",
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 4.5], gridcolor="lightgrey", linecolor="black"
            )
        ),
        showlegend=True,
        title="Satisfaction Overlap (Scale 1-4)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width="stretch")

    # ==========================================
    # SECTION 7: BUSINESS IMPACT
    # ==========================================
    st.subheader("💼 Business Impact & Priority")

    if prob_A * cost_A > prob_B * cost_B:
        st.error(
            f"**Priority: Employee A**\nHigher blended financial and attrition risk exposure (₹{prob_A * cost_A:,.2f})."
        )
    else:
        st.error(
            f"**Priority: Employee B**\nHigher blended financial and attrition risk exposure (₹{prob_B * cost_B:,.2f})."
        )

    # ==========================================
    # SECTION 10: EXPORT
    # ==========================================
    st.markdown("---")
    st.subheader("📥 Export Center")

    export_dict = {
        "Employee_A": std_A,
        "Employee_A_Risk": prob_A,
        "Employee_B": std_B,
        "Employee_B_Risk": prob_B,
    }

    json_data = json.dumps(export_dict, indent=4)
    st.download_button(
        "📜 Download JSON",
        data=json_data,
        file_name=f"employee_attrition_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )

    def generate_comp_pdf(std_A, prob_A, std_B, prob_B, summary_text):
        buffer = BytesIO()

        def add_footer(canvas, doc):
            canvas.saveState()
            canvas.setFont("Helvetica", 9)
            canvas.setFillColor(colors.gray)
            canvas.drawString(
                inch,
                0.75 * inch,
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            )
            canvas.drawRightString(7.5 * inch, 0.75 * inch, f"Page {doc.page}")
            canvas.restoreState()

        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=inch,
            leftMargin=inch,
            topMargin=inch,
            bottomMargin=inch,
        )

        elements = []
        styles = getSampleStyleSheet()
        title_style = styles["Title"]
        h2_style = styles["Heading2"]
        normal_style = styles["Normal"]

        elements.append(Paragraph("Enterprise Employee Comparison Report", title_style))
        elements.append(Spacer(1, 12))
        elements.append(
            Paragraph(
                "Executive insights detailing comparative attrition risk and retention strategies.",
                normal_style,
            )
        )
        elements.append(Spacer(1, 24))

        elements.append(Paragraph("Executive Summary", h2_style))
        for s in summary_text:
            elements.append(Paragraph(f"• {s}", normal_style))
            elements.append(Spacer(1, 6))

        elements.append(Spacer(1, 12))

        elements.append(
            Paragraph(f"<b>Employee A Risk:</b> {prob_A*100:.1f}%", normal_style)
        )
        elements.append(
            Paragraph(f"<b>Employee B Risk:</b> {prob_B*100:.1f}%", normal_style)
        )

        doc.build(elements, onFirstPage=add_footer, onLaterPages=add_footer)
        buffer.seek(0)
        return buffer.read()

    try:
        pdf_bytes = generate_comp_pdf(std_A, prob_A, std_B, prob_B, summary)
        st.download_button(
            "📑 Download Executive PDF",
            data=pdf_bytes,
            file_name=f"employee_attrition_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.error("PDF generation failed.")

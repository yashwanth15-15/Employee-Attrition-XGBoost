import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import shap
from datetime import datetime
from io import BytesIO
import json
import base64

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_predictions
from helpers.hr_recommendation_engine import generate_hr_recommendation
from helpers.risk_calculator import calculate_risk

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

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
    st.error("❌ Model artifacts not found. Please ensure models exist in the 'models/' directory.")
    st.stop()

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def predict_employee(employee_dict):
    """Predict and explain risk for a single employee dict."""
    df_emp = pd.DataFrame([employee_dict])
    encoded_df = df_emp.copy()
    
    # Handle missing defaults
    if 'PerformanceRating' not in encoded_df.columns:
        encoded_df['PerformanceRating'] = 3
    if 'YearsWithCurrManager' not in encoded_df.columns:
        encoded_df['YearsWithCurrManager'] = encoded_df.get('YearsInCurrentRole', 0)
    if 'JobInvolvement' not in encoded_df.columns:
        encoded_df['JobInvolvement'] = 3
    if 'Education' not in encoded_df.columns:
        encoded_df['Education'] = 3
    if 'NumCompaniesWorked' not in encoded_df.columns:
        encoded_df['NumCompaniesWorked'] = 1
    if 'RelationshipSatisfaction' not in encoded_df.columns:
        encoded_df['RelationshipSatisfaction'] = 3
    if 'JobLevel' not in encoded_df.columns:
        encoded_df['JobLevel'] = 1
    if 'PercentSalaryHike' not in encoded_df.columns:
        encoded_df['PercentSalaryHike'] = 11

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
        "Training Times Last Year": "TrainingTimesLastYear"
    }
    encoded_df.rename(columns=rename_map, inplace=True)
    
    # Ensure all feature names exist
    for col in feature_names:
        if col not in encoded_df.columns:
            encoded_df[col] = 0

    for col, encoder in encoders.items():
        if col == "Attrition": continue
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
        "Age": int(employee_dict.get("Age", 30)),
        "Gender": employee_dict.get("Gender", "Male"),
        "Department": employee_dict.get("Department", "Sales"),
        "Monthly Income": float(employee_dict.get("Monthly Income", 5000) or employee_dict.get("MonthlyIncome", 5000)),
        "Marital Status": employee_dict.get("Marital Status", "Single"),
        "OverTime": employee_dict.get("OverTime", "No"),
        "Years At Company": int(employee_dict.get("Years At Company", 0)),
        "Total Working Years": int(employee_dict.get("Total Working Years", 0)),
        "Job Satisfaction": int(employee_dict.get("Job Satisfaction", 3)),
        "Environment Satisfaction": int(employee_dict.get("Environment Satisfaction", 3)),
        "Work Life Balance": int(employee_dict.get("Work Life Balance", 3)),
        "Years Since Last Promotion": int(employee_dict.get("Years Since Last Promotion", 0)),
        "Training Times Last Year": int(employee_dict.get("Training Times Last Year", 2)),
        "Business Travel": employee_dict.get("Business Travel", "Non-Travel"),
        "Distance From Home": int(employee_dict.get("Distance From Home", 5)),
        "Performance Rating": 3,
        "Stock Option Level": int(employee_dict.get("Stock Option Level", 0)),
        "Years In Current Role": int(employee_dict.get("Years In Current Role", 0)),
    }
    
    rec = generate_hr_recommendation(standard_dict, prob, risk)
    return prob, risk, shap_vals, rec, standard_dict, X

# ==========================================
# HEADER
# ==========================================
st.title("👥 Employee Comparison")
st.markdown("Side-by-side executive analysis of employee retention risks and actionable insights.")
st.markdown("---")

# ==========================================
# SECTION 1: DATA SELECTION
# ==========================================
tab_db, tab_csv, tab_manual = st.tabs(["🗄️ Select from Database", "📁 Upload CSV", "✍️ Manual Entry"])

emp_A_data = None
emp_B_data = None
ready_to_compare = False

with tab_db:
    db_df = get_predictions()
    if db_df.empty:
        st.info("No predictions found in the database.")
    else:
        db_df['label'] = "ID: " + db_df['id'].astype(str) + " - " + db_df['department'] + " (" + db_df['prediction_date'].str.split(' ').str[0] + ")"
        
        col1, col2 = st.columns(2)
        with col1:
            sel_A = st.selectbox("Select Employee A", db_df['label'].tolist(), key="db_sel_a")
        with col2:
            sel_B = st.selectbox("Select Employee B", db_df['label'].tolist(), key="db_sel_b", index=min(1, len(db_df)-1))
            
        if st.button("Compare Database Employees", type="primary"):
            row_A = db_df[db_df['label'] == sel_A].iloc[0].to_dict()
            row_B = db_df[db_df['label'] == sel_B].iloc[0].to_dict()
            
            # Map DB columns back to standard names
            def map_db_to_standard(row):
                return {
                    "Age": row.get("age", 30),
                    "Gender": row.get("gender", "Unknown"),
                    "Department": row.get("department", "Unknown"),
                    "Monthly Income": row.get("monthly_income", 5000),
                    "Marital Status": row.get("marital_status", "Single"),
                    "OverTime": row.get("overtime", "No"),
                    "Years At Company": row.get("years_at_company", 0),
                    "Total Working Years": row.get("years_at_company", 0) + 2, # Approximation if missing
                    "Job Satisfaction": row.get("job_satisfaction", 3),
                    "Environment Satisfaction": row.get("environment_satisfaction", 3),
                    "Work Life Balance": row.get("work_life_balance", 3),
                    "Business Travel": "Non-Travel",
                    "Stock Option Level": 0,
                    "Training Times Last Year": 2,
                    "Years Since Last Promotion": 0,
                    "Years In Current Role": 0,
                    "Distance From Home": 5
                }
                
            emp_A_data = map_db_to_standard(row_A)
            emp_B_data = map_db_to_standard(row_B)
            ready_to_compare = True

with tab_csv:
    st.write("Upload a CSV file containing exactly two rows representing Employee A and Employee B.")
    
    # Template download
    template_df = pd.DataFrame([{
        "Age": 30, "Gender": "Male", "Department": "Sales", "Monthly Income": 5000,
        "Marital Status": "Single", "OverTime": "No", "Years At Company": 5, "Total Working Years": 5,
        "Job Satisfaction": 3, "Environment Satisfaction": 3, "Work Life Balance": 3,
        "Years Since Last Promotion": 0, "Training Times Last Year": 2, "Business Travel": "Travel_Rarely",
        "Distance From Home": 5, "Stock Option Level": 1, "Years In Current Role": 3
    }])
    csv_template = template_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV Template", data=csv_template, file_name="comparison_template.csv", mime="text/csv")
    
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
        a_dept = st.selectbox("Department (A)", ["Sales", "Research & Development", "Human Resources"], key="a_dept")
        a_income = st.number_input("Monthly Income (A)", 1000, 100000, 5000, key="a_inc")
        a_ot = st.selectbox("OverTime (A)", ["Yes", "No"], key="a_ot")
        a_js = st.slider("Job Satisfaction (A)", 1, 4, 3, key="a_js")
    with m_col2:
        st.subheader("Employee B")
        b_age = st.number_input("Age (B)", 18, 65, 35, key="b_age")
        b_dept = st.selectbox("Department (B)", ["Sales", "Research & Development", "Human Resources"], key="b_dept", index=1)
        b_income = st.number_input("Monthly Income (B)", 1000, 100000, 8000, key="b_inc")
        b_ot = st.selectbox("OverTime (B)", ["Yes", "No"], key="b_ot", index=1)
        b_js = st.slider("Job Satisfaction (B)", 1, 4, 4, key="b_js")
        
    if st.button("Compare Manual Entries", type="primary"):
        emp_A_data = {"Age": a_age, "Department": a_dept, "Monthly Income": a_income, "OverTime": a_ot, "Job Satisfaction": a_js}
        emp_B_data = {"Age": b_age, "Department": b_dept, "Monthly Income": b_income, "OverTime": b_ot, "Job Satisfaction": b_js}
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
    if prob_A > prob_B + 0.15:
        summary.append(f"Employee A has a significantly higher attrition risk ({prob_A*100:.1f}%) compared to Employee B ({prob_B*100:.1f}%).")
        if std_A['OverTime'] == 'Yes' and std_B['OverTime'] == 'No':
            summary.append("Employee A's frequent overtime is a major distinguishing risk factor.")
    elif prob_B > prob_A + 0.15:
        summary.append(f"Employee B has a significantly higher attrition risk ({prob_B*100:.1f}%) compared to Employee A ({prob_A*100:.1f}%).")
    else:
        summary.append(f"Both employees exhibit similar retention profiles (A: {prob_A*100:.1f}%, B: {prob_B*100:.1f}%).")
        
    if cost_A > cost_B and prob_A > 0.4:
        summary.append(f"Immediate intervention is recommended for Employee A due to the high replacement exposure of ₹{cost_A:,.2f}.")
    elif cost_B > cost_A and prob_B > 0.4:
        summary.append(f"Immediate intervention is recommended for Employee B due to the high replacement exposure of ₹{cost_B:,.2f}.")
        
    st.info(" ".join(summary))
    st.markdown("---")
    
    # ==========================================
    # SECTION 3: HEAD-TO-HEAD COMPARISON
    # ==========================================
    st.subheader("🔍 Feature Comparison")
    
    keys = ["Age", "Department", "Monthly Income", "OverTime", "Job Satisfaction", "Work Life Balance", "Years At Company"]
    comp_rows = []
    for k in keys:
        comp_rows.append({"Metric": k, "Employee A": std_A.get(k), "Employee B": std_B.get(k)})
        
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
    
    # ==========================================
    # SECTION 4 & 5: SHAP COMPARISON
    # ==========================================
    st.subheader("🔥 Risk Driver Comparison (SHAP)")
    
    def get_top_shap_df(shap_v, feats):
        abs_s = np.abs(shap_v[0])
        top_idx = np.argsort(abs_s)[::-1][:5]
        return pd.DataFrame({
            "Feature": [feats[i] for i in top_idx],
            "Contribution": [shap_v[0][i] * 100 for i in top_idx]
        })
        
    top_A = get_top_shap_df(shap_A, feature_names)
    top_B = get_top_shap_df(shap_B, feature_names)
    
    sh1, sh2 = st.columns(2)
    with sh1:
        fig_A = px.bar(top_A, x='Contribution', y='Feature', orientation='h', title='Employee A Top Risk Drivers')
        fig_A.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_A, use_container_width=True)
    with sh2:
        fig_B = px.bar(top_B, x='Contribution', y='Feature', orientation='h', title='Employee B Top Risk Drivers')
        fig_B.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_B, use_container_width=True)

    # ==========================================
    # SECTION 6: RECOMMENDATION COMPARISON
    # ==========================================
    st.subheader("🎯 Recommendation Matrix")
    
    common_immediate = list(set(rec_A['immediate_actions']) & set(rec_B['immediate_actions']))
    if common_immediate:
        st.success("**Common Actions Required:** " + ", ".join(common_immediate))
        
    r1, r2 = st.columns(2)
    with r1:
        with st.container(border=True):
            st.markdown("### Employee A")
            for act in rec_A['immediate_actions']: st.write(f"🔴 {act}")
            for act in rec_A['medium_term_actions']: st.write(f"🟡 {act}")
            for act in rec_A['long_term_strategy']: st.write(f"🟢 {act}")
            
    with r2:
        with st.container(border=True):
            st.markdown("### Employee B")
            for act in rec_B['immediate_actions']: st.write(f"🔴 {act}")
            for act in rec_B['medium_term_actions']: st.write(f"🟡 {act}")
            for act in rec_B['long_term_strategy']: st.write(f"🟢 {act}")
            
    st.markdown("---")

    # ==========================================
    # SECTION 8: VISUAL ANALYTICS (RADAR)
    # ==========================================
    st.subheader("🕸️ Cultural Fit & Satisfaction Radar")
    
    categories = ['Job Satisfaction', 'Environment Satisfaction', 'Work Life Balance', 'Relationship Satisfaction']
    
    def get_radar_vals(std_dict):
        return [
            std_dict.get('Job Satisfaction', 3),
            std_dict.get('Environment Satisfaction', 3),
            std_dict.get('Work Life Balance', 3),
            3 # Default Relationship
        ]
        
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=get_radar_vals(std_A), theta=categories, fill='toself', name='Employee A'))
    fig.add_trace(go.Scatterpolar(r=get_radar_vals(std_B), theta=categories, fill='toself', name='Employee B'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 4])), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================
    # SECTION 7: BUSINESS IMPACT
    # ==========================================
    st.subheader("💼 Business Impact & Priority")
    
    if prob_A * cost_A > prob_B * cost_B:
        st.error(f"**Priority: Employee A**\nHigher blended financial and attrition risk exposure (₹{prob_A * cost_A:,.2f}).")
    else:
        st.error(f"**Priority: Employee B**\nHigher blended financial and attrition risk exposure (₹{prob_B * cost_B:,.2f}).")

    # ==========================================
    # SECTION 10: EXPORT
    # ==========================================
    st.markdown("---")
    st.subheader("📥 Export Center")
    
    export_dict = {
        "Employee_A": std_A,
        "Employee_A_Risk": prob_A,
        "Employee_B": std_B,
        "Employee_B_Risk": prob_B
    }
    
    json_data = json.dumps(export_dict, indent=4)
    st.download_button("📜 Download JSON", data=json_data, file_name="employee_comparison.json", mime="application/json")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# Import database function
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_predictions

# For PDF export
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

st.set_page_config(page_title="Retention Cost Planning", layout="wide", page_icon="💰")

# ==========================================
# HEADER
# ==========================================
st.title("💰 Retention Cost & Workforce Planning")
st.markdown("Executive financial dashboard estimating organizational attrition cost and retention program ROI.")

# ==========================================
# FETCH DATA
# ==========================================
@st.cache_data(ttl=60)
def load_data():
    df = get_predictions()
    if not df.empty:
        df['prediction_date'] = pd.to_datetime(df['prediction_date'])
        df['prediction_probability'] = pd.to_numeric(df['prediction_probability'])
        df['monthly_income'] = pd.to_numeric(df['monthly_income'])
        df['replacement_cost'] = pd.to_numeric(df['replacement_cost'])
        df['job_satisfaction'] = pd.to_numeric(df['job_satisfaction'])
        df['work_life_balance'] = pd.to_numeric(df['work_life_balance'])
    return df

df = load_data()

if df.empty:
    st.info("ℹ️ No employee predictions available yet. Please generate predictions to view financial analytics.")
    st.stop()

# ==========================================
# CORE FINANCIAL CALCULATIONS
# ==========================================
total_workforce = len(df)
high_risk_df = df[df['risk_category'] == 'High']
high_risk_count = len(high_risk_df)

total_replacement_cost = df['replacement_cost'].sum()
annual_attrition_cost = high_risk_df['replacement_cost'].sum()
avg_replacement_cost = df['replacement_cost'].mean() if total_workforce > 0 else 0

dept_costs = df.groupby('department')['replacement_cost'].sum().reset_index()
highest_cost_dept = dept_costs.sort_values('replacement_cost', ascending=False).iloc[0]['department'] if not dept_costs.empty else "N/A"

# ==========================================
# SECTION 1: EXECUTIVE KPIs
# ==========================================
st.markdown("### 🏦 Executive Financial Summary")

k1, k2, k3, k4 = st.columns(4)
k1.metric("👥 Total Workforce", total_workforce)
k2.metric("🔴 High Risk Employees", high_risk_count)
k3.metric("💸 Total Org Replacement Cost", f"₹{total_replacement_cost:,.2f}")
k4.metric("🚨 Potential Annual Attrition Cost", f"₹{annual_attrition_cost:,.2f}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("📊 Average Replacement Cost", f"₹{avg_replacement_cost:,.2f}")
c2.metric("🏢 Highest Cost Dept", highest_cost_dept)
c3.metric("🎯 Recommended Retention Budget", f"₹{annual_attrition_cost * 0.10:,.2f}")
c4.metric("📈 Baseline ROI Potential", "High")

st.markdown("---")

# ==========================================
# SECTION 3: WHAT-IF BUSINESS SCENARIOS
# ==========================================
st.subheader("🧪 What-If Business Scenarios")
st.write("Simulate the financial impact of running a retention program that successfully reduces high-risk attrition.")

sim_col1, sim_col2 = st.columns([1, 2])

with sim_col1:
    with st.container(border=True):
        st.markdown("#### Program Success Rate")
        success_rate = st.slider("Target Attrition Reduction (%)", min_value=0, max_value=50, step=5, value=15)
        
        retention_budget = annual_attrition_cost * 0.10
        st.caption(f"Assuming a budget of **₹{retention_budget:,.2f}** (10% of total risk exposure).")

with sim_col2:
    savings = annual_attrition_cost * (success_rate / 100)
    remaining_cost = annual_attrition_cost - savings
    roi = ((savings - retention_budget) / retention_budget) * 100 if retention_budget > 0 else 0
    
    r1, r2, r3 = st.columns(3)
    with st.container(border=True):
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Estimated Savings (Cost Avoided)", f"₹{savings:,.2f}")
        sc2.metric("Remaining Financial Exposure", f"₹{remaining_cost:,.2f}")
        
        if roi > 0:
            sc3.metric("Estimated Program ROI", f"+{roi:,.1f}%")
        else:
            sc3.metric("Estimated Program ROI", f"{roi:,.1f}%", delta_color="inverse")

st.markdown("---")

# ==========================================
# SECTION 2: FINANCIAL ANALYTICS
# ==========================================
st.subheader("📊 Financial Analytics")

v1, v2 = st.columns(2)
with v1:
    # Replacement Cost by Department
    fig_bar = px.bar(dept_costs, x='department', y='replacement_cost', color='replacement_cost', 
                     color_continuous_scale='Reds', title='Total Financial Exposure by Department')
    fig_bar.update_layout(yaxis_title="Replacement Cost (₹)", xaxis_title="Department")
    st.plotly_chart(fig_bar, use_container_width=True)

with v2:
    # Attrition Cost Distribution (Pie)
    if not high_risk_df.empty:
        high_dept_costs = high_risk_df.groupby('department')['replacement_cost'].sum().reset_index()
        fig_pie = px.pie(high_dept_costs, names='department', values='replacement_cost', hole=0.4, 
                         title="High-Risk Cost Breakdown by Department")
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No high-risk cost distribution available.")

v3, v4 = st.columns(2)
with v3:
    # Salary vs Replacement Cost
    fig_scatter = px.scatter(df, x='monthly_income', y='replacement_cost', color='risk_category',
                             color_discrete_map={'High': 'salmon', 'Medium': 'gold', 'Low': 'lightgreen'}, 
                             title='Salary vs Replacement Cost')
    fig_scatter.update_layout(xaxis_title="Monthly Income (₹)", yaxis_title="Replacement Cost (₹)")
    st.plotly_chart(fig_scatter, use_container_width=True)

with v4:
    # Risk Categories Financial Weight
    risk_weights = df.groupby('risk_category')['replacement_cost'].sum().reset_index()
    fig_weight = px.bar(risk_weights, x='risk_category', y='replacement_cost', color='risk_category',
                        color_discrete_map={'High': 'salmon', 'Medium': 'gold', 'Low': 'lightgreen'},
                        title='Financial Weight by Risk Category')
    fig_weight.update_layout(yaxis_title="Total Replacement Cost (₹)")
    st.plotly_chart(fig_weight, use_container_width=True)

st.markdown("---")

# ==========================================
# SECTION 5: PRIORITY MATRIX
# ==========================================
st.subheader("🏆 Retention Priority Matrix")

dept_metrics = df.groupby('department').agg(
    Employee_Count=('id', 'count'),
    Financial_Risk=('replacement_cost', lambda x: x[df.loc[x.index, 'risk_category'] == 'High'].sum()),
    High_Risk_Count=('risk_category', lambda x: (x == 'High').sum()),
    Avg_Risk=('prediction_probability', 'mean')
).reset_index()

dept_metrics['Retention_Priority'] = dept_metrics['Financial_Risk'].rank(ascending=False, method='min').astype(int)
dept_metrics = dept_metrics.sort_values('Retention_Priority')

st.dataframe(
    dept_metrics.rename(columns={
        'department': 'Department',
        'Employee_Count': 'Total Headcount',
        'Financial_Risk': 'Financial Risk (₹)',
        'High_Risk_Count': 'High Risk Count',
        'Avg_Risk': 'Avg Risk Probability',
        'Retention_Priority': 'Priority Rank'
    }).style.format({'Financial Risk (₹)': '₹{:,.2f}', 'Avg Risk Probability': '{:.2%}'}),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# ==========================================
# SECTION 4 & 6: BUDGET PLANNING & INSIGHTS
# ==========================================
st.subheader("💡 Executive Insights & Budget Planning")

b_col1, b_col2 = st.columns([2, 1])

with b_col1:
    with st.container(border=True):
        st.markdown("#### 🧠 Auto-Generated Financial Insights")
        
        insights = []
        if total_replacement_cost > 0:
            high_risk_pct = (annual_attrition_cost / total_replacement_cost) * 100
            insights.append(f"High-risk employees account for **{high_risk_pct:.1f}%** of the total organizational replacement value.")
            
        if not dept_metrics.empty:
            top_dept = dept_metrics.iloc[0]
            if top_dept['Financial_Risk'] > 0:
                insights.append(f"**{top_dept['department']}** is the highest priority department, contributing **₹{top_dept['Financial_Risk']:,.2f}** to projected attrition costs.")
                
        ot_high = df[(df['overtime'] == 'Yes') & (df['risk_category'] == 'High')]
        if not ot_high.empty:
            ot_cost = ot_high['replacement_cost'].sum()
            insights.append(f"Reducing overtime for high-risk profiles could intercept up to **₹{ot_cost:,.2f}** in attrition exposure.")
            
        insights.append(f"An intervention achieving a {success_rate}% success rate will save approximately **₹{savings:,.2f}** annually.")
        
        for insight in insights:
            st.info(f"💡 {insight}")

with b_col2:
    with st.container(border=True):
        st.markdown("#### 💰 Suggested Budget Allocation")
        st.caption(f"Based on 10% of total exposed risk (₹{retention_budget:,.2f})")
        
        b1, b2 = st.columns([3, 1])
        b1.write("🏆 Rewards & Bonuses (40%)")
        b2.write(f"₹{retention_budget * 0.40:,.0f}")
        
        b1.write("🎓 Training & Upskilling (30%)")
        b2.write(f"₹{retention_budget * 0.30:,.0f}")
        
        b1.write("🤝 Recruiting Padding (20%)")
        b2.write(f"₹{retention_budget * 0.20:,.0f}")
        
        b1.write("🎉 Employee Engagement (10%)")
        b2.write(f"₹{retention_budget * 0.10:,.0f}")

st.markdown("---")

# ==========================================
# SECTION 7: EXPORTS
# ==========================================
st.subheader("📥 Export Center")
ex_col1, ex_col2, ex_col3 = st.columns(3)

csv_data = dept_metrics.to_csv(index=False).encode('utf-8')
with ex_col1:
    st.download_button("📄 Download Priority CSV", data=csv_data, file_name="retention_priority.csv", mime="text/csv", width="stretch")

json_data = dept_metrics.to_json(orient='records')
with ex_col2:
    st.download_button("📜 Download Priority JSON", data=json_data, file_name="retention_priority.json", mime="application/json", width="stretch")

# PDF Generation
def generate_financial_pdf(dept_metrics_df, total_cost, target_savings):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []
    styles = getSampleStyleSheet()
    
    elements.append(Paragraph("Executive Financial Retention Report", styles['Title']))
    elements.append(Spacer(1, 12))
    
    elements.append(Paragraph(f"Total Potential Attrition Cost: INR {total_cost:,.2f}", styles['Normal']))
    elements.append(Paragraph(f"Target Savings at {success_rate}% success: INR {target_savings:,.2f}", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    data = [dept_metrics_df.columns.tolist()] + dept_metrics_df.round(2).values.tolist()
    
    t = Table(data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    elements.append(t)
    doc.build(elements)
    buffer.seek(0)
    return buffer.read()

try:
    pdf_bytes = generate_financial_pdf(dept_metrics, annual_attrition_cost, savings)
    with ex_col3:
        st.download_button("📑 Download Financial PDF", data=pdf_bytes, file_name="Financial_Retention_Report.pdf", mime="application/pdf", width="stretch")
except Exception as e:
    with ex_col3:
        st.error("PDF generation failed.")

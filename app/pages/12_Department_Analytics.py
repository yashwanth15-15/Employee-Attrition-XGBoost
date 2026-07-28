import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
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

st.set_page_config(page_title="Department Analytics", layout="wide", page_icon="📊")

# ==========================================
# HEADER
# ==========================================
st.title("📊 Department Analytics Dashboard")
st.markdown("Enterprise HR analytics tracking real-time attrition risk across all departments.")

# ==========================================
# FETCH DATA
# ==========================================
@st.cache_data(ttl=60) # Cache but refresh frequently
def load_data():
    df = get_predictions()
    if not df.empty:
        # Convert types
        df['prediction_date'] = pd.to_datetime(df['prediction_date'])
        df['prediction_probability'] = pd.to_numeric(df['prediction_probability'])
        df['monthly_income'] = pd.to_numeric(df['monthly_income'])
        df['replacement_cost'] = pd.to_numeric(df['replacement_cost'])
        df['job_satisfaction'] = pd.to_numeric(df['job_satisfaction'])
        df['environment_satisfaction'] = pd.to_numeric(df['environment_satisfaction'])
        df['work_life_balance'] = pd.to_numeric(df['work_life_balance'])
    return df

raw_df = load_data()

if raw_df.empty:
    st.info("ℹ️ No employee predictions available yet. Please generate predictions to view analytics.")
    st.stop()

# ==========================================
# SIDEBAR FILTERS
# ==========================================
st.sidebar.header("🔍 Interactive Filters")

departments = raw_df['department'].unique().tolist()
selected_deps = st.sidebar.multiselect("Department", departments, default=departments)

genders = raw_df['gender'].unique().tolist()
selected_gender = st.sidebar.multiselect("Gender", genders, default=genders)

marital = raw_df['marital_status'].unique().tolist()
selected_marital = st.sidebar.multiselect("Marital Status", marital, default=marital)

travel = raw_df['business_travel'].unique().tolist() if 'business_travel' in raw_df.columns else []
if travel:
    selected_travel = st.sidebar.multiselect("Business Travel", travel, default=travel)

overtime_opts = raw_df['overtime'].unique().tolist()
selected_ot = st.sidebar.multiselect("Overtime", overtime_opts, default=overtime_opts)

risks = raw_df['risk_category'].unique().tolist()
selected_risk = st.sidebar.multiselect("Risk Level", risks, default=risks)

# Apply filters
df = raw_df[
    (raw_df['department'].isin(selected_deps)) &
    (raw_df['gender'].isin(selected_gender)) &
    (raw_df['marital_status'].isin(selected_marital)) &
    (raw_df['overtime'].isin(selected_ot)) &
    (raw_df['risk_category'].isin(selected_risk))
]

if travel:
    df = df[df['business_travel'].isin(selected_travel)]

if df.empty:
    st.warning("⚠️ No data matches the selected filters.")
    st.stop()

# ==========================================
# SECTION 1: EXECUTIVE KPIs
# ==========================================
total_emp = len(df)
high_risk = len(df[df['risk_category'] == 'High'])
med_risk = len(df[df['risk_category'] == 'Medium'])
low_risk = len(df[df['risk_category'] == 'Low'])

avg_risk = df['prediction_probability'].mean() * 100

dep_risk_avg = df.groupby('department')['prediction_probability'].mean().reset_index()
highest_risk_dep = dep_risk_avg.sort_values('prediction_probability', ascending=False).iloc[0]['department']
lowest_risk_dep = dep_risk_avg.sort_values('prediction_probability', ascending=True).iloc[0]['department']

today = datetime.now().date()
preds_today = len(df[df['prediction_date'].dt.date == today])

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("👥 Total Employees", total_emp)
    st.metric("📈 Highest Risk Dept", highest_risk_dep)
with col2:
    st.metric("🔴 High Risk Employees", high_risk)
    st.metric("📉 Lowest Risk Dept", lowest_risk_dep)
with col3:
    st.metric("🟡 Medium Risk Employees", med_risk)
    st.metric("⚡ Average Attrition Risk", f"{avg_risk:.1f}%")
with col4:
    st.metric("🟢 Low Risk Employees", low_risk)
    st.metric("📅 Predictions Today", preds_today)

st.markdown("---")

# ==========================================
# SECTION 2: DEPARTMENT OVERVIEW TABLE
# ==========================================
st.subheader("🏢 Department Overview")

dept_summary = df.groupby('department').agg(
    Employees=('id', 'count'),
    Average_Risk=('prediction_probability', lambda x: x.mean() * 100),
    High_Risk_Count=('risk_category', lambda x: (x == 'High').sum()),
    Average_Salary=('monthly_income', 'mean'),
    Avg_Job_Sat=('job_satisfaction', 'mean'),
    Avg_Env_Sat=('environment_satisfaction', 'mean'),
    Avg_WLB=('work_life_balance', 'mean'),
    Total_Replacement_Cost=('replacement_cost', 'sum')
).reset_index()

dept_summary = dept_summary.sort_values('Average_Risk', ascending=False).round(2)

st.dataframe(
    dept_summary.rename(columns={
        'department': 'Department',
        'Average_Risk': 'Avg Risk (%)',
        'High_Risk_Count': 'High Risk Emp',
        'Average_Salary': 'Avg Salary (₹)',
        'Avg_Job_Sat': 'Job Sat (1-4)',
        'Avg_Env_Sat': 'Env Sat (1-4)',
        'Avg_WLB': 'Work-Life (1-4)',
        'Total_Replacement_Cost': 'Replacement Cost (₹)'
    }),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# ==========================================
# SECTION 5: RISK DISTRIBUTION
# ==========================================
st.subheader("⚖️ Risk Distribution")

r_col1, r_col2 = st.columns([1, 2])
with r_col1:
    # Gauge Chart
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = avg_risk,
        title = {'text': "Average Org Risk (%)"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps' : [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "gold"},
                {'range': [70, 100], 'color': "salmon"}
            ],
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

with r_col2:
    # Risk Distribution Pie
    risk_counts = df['risk_category'].value_counts().reset_index()
    risk_counts.columns = ['Risk Category', 'Count']
    color_map = {'High': 'salmon', 'Medium': 'gold', 'Low': 'lightgreen'}
    fig_pie = px.pie(risk_counts, names='Risk Category', values='Count', hole=0.4, 
                     color='Risk Category', color_discrete_map=color_map,
                     title="Risk Category Distribution")
    fig_pie.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# ==========================================
# SECTION 4: VISUAL ANALYTICS
# ==========================================
st.subheader("📈 Visual Analytics")

tab1, tab2, tab3 = st.tabs(["Department Metrics", "Correlations", "Trends & Costs"])

with tab1:
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        # Department Risk Bar
        fig_bar = px.bar(dept_summary, x='department', y='Average_Risk', color='Average_Risk', 
                         color_continuous_scale='Reds', title='Average Risk by Department (%)')
        st.plotly_chart(fig_bar, use_container_width=True)
    with v_col2:
        # Department Satisfaction Heatmap
        heat_data = dept_summary[['department', 'Avg_Job_Sat', 'Avg_Env_Sat', 'Avg_WLB']].set_index('department')
        fig_heat = px.imshow(heat_data.T, text_auto=True, aspect="auto", 
                             color_continuous_scale='Blues', title='Satisfaction Metrics by Dept (1-4)')
        st.plotly_chart(fig_heat, use_container_width=True)

with tab2:
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        # Salary vs Risk Scatter
        fig_scatter = px.scatter(df, x='monthly_income', y='prediction_probability', color='risk_category',
                                 color_discrete_map=color_map, title='Monthly Income vs Attrition Probability')
        st.plotly_chart(fig_scatter, use_container_width=True)
    with c_col2:
        # Overtime vs Risk Stacked Bar
        ot_risk = df.groupby(['overtime', 'risk_category']).size().reset_index(name='Count')
        fig_ot = px.bar(ot_risk, x='overtime', y='Count', color='risk_category', 
                        color_discrete_map=color_map, title='Overtime vs Attrition Risk', barmode='stack')
        st.plotly_chart(fig_ot, use_container_width=True)

with tab3:
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        # Monthly Trend
        df['Month'] = df['prediction_date'].dt.to_period('M').astype(str)
        trend = df.groupby('Month').size().reset_index(name='Predictions')
        fig_trend = px.line(trend, x='Month', y='Predictions', markers=True, title='Monthly Prediction Trend')
        st.plotly_chart(fig_trend, use_container_width=True)
    with t_col2:
        # Department Replacement Cost
        fig_cost = px.bar(dept_summary, x='Total_Replacement_Cost', y='department', orientation='h',
                          title='Total Replacement Cost Exposure by Dept (₹)', color='Total_Replacement_Cost', color_continuous_scale='Reds')
        st.plotly_chart(fig_cost, use_container_width=True)

# Top 10 High Risk
st.markdown("#### 🚨 Top 10 High Risk Employees (by Database ID)")
high_risk_df = df[df['risk_category'] == 'High'].sort_values('prediction_probability', ascending=False).head(10)
if not high_risk_df.empty:
    fig_top10 = px.bar(high_risk_df, x='id', y='prediction_probability', text_auto='.2f',
                       title='Top 10 High Risk Profiles', labels={'id': 'Database ID', 'prediction_probability': 'Probability'})
    fig_top10.update_xaxes(type='category')
    st.plotly_chart(fig_top10, use_container_width=True)
else:
    st.info("No high-risk employees found in the current filtered dataset.")

st.markdown("---")

# ==========================================
# SECTION 6, 7 & 8: BUSINESS IMPACT & INSIGHTS
# ==========================================
st.subheader("💼 Business Impact & Executive Insights")

b_col1, b_col2 = st.columns([1, 2])

with b_col1:
    # Calculate Business Impact
    total_cost_exposure = high_risk_df['replacement_cost'].sum() if not high_risk_df.empty else 0
    highest_cost_dept = dept_summary.sort_values('Total_Replacement_Cost', ascending=False).iloc[0]
    
    with st.container(border=True):
        st.markdown("### Financial Exposure")
        st.metric("Estimated Total Replacement Cost (High Risk)", f"₹{total_cost_exposure:,.2f}")
        st.metric("Highest Cost Department", highest_cost_dept['department'])
        st.metric(f"Exposure in {highest_cost_dept['department']}", f"₹{highest_cost_dept['Total_Replacement_Cost']:,.2f}")

with b_col2:
    with st.container(border=True):
        st.markdown("### 🧠 Auto-Generated Insights")
        
        insights = []
        insights.append(f"**{highest_risk_dep}** has the highest average attrition risk at **{dept_summary.iloc[0]['Average_Risk']:.1f}%**.")
        insights.append(f"**{lowest_risk_dep}** has the best retention outlook.")
        
        # Check overtime correlation
        ot_high = len(df[(df['overtime'] == 'Yes') & (df['risk_category'] == 'High')])
        non_ot_high = len(df[(df['overtime'] == 'No') & (df['risk_category'] == 'High')])
        if ot_high > non_ot_high:
            insights.append("Overtime strongly correlates with high attrition in the current population.")
            
        # Check WLB
        lowest_wlb = dept_summary.sort_values('Avg_WLB').iloc[0]
        if lowest_wlb['Avg_WLB'] < 2.5:
            insights.append(f"**{lowest_wlb['department']}** shows declining work-life balance (Avg: {lowest_wlb['Avg_WLB']:.1f}).")
            
        for insight in insights:
            st.info(f"💡 {insight}")

st.markdown("### 📋 Executive Recommendations")
e_col1, e_col2, e_col3 = st.columns(3)

with e_col1:
    with st.container(border=True):
        st.markdown("🔴 **Immediate**")
        st.write(f"✔ Investigate high risk drivers in {highest_risk_dep}")
        if total_cost_exposure > 0:
            st.write("✔ Review retention strategies for critical profiles")

with e_col2:
    with st.container(border=True):
        st.markdown("🟡 **Medium**")
        if ot_high > non_ot_high:
            st.write("✔ Reduce overtime workload across departments")
        st.write("✔ Review compensation strategy")

with e_col3:
    with st.container(border=True):
        st.markdown("🟢 **Long-Term**")
        st.write("✔ Improve promotion and career growth opportunities")
        st.write(f"✔ Replicate retention success of {lowest_risk_dep} across org")

st.markdown("---")

# ==========================================
# SECTION 9: EXPORTS
# ==========================================
st.subheader("📥 Export Center")
ex_col1, ex_col2, ex_col3 = st.columns(3)

csv_data = df.to_csv(index=False).encode('utf-8')
with ex_col1:
    st.download_button("📄 Download CSV", data=csv_data, file_name="department_analytics.csv", mime="text/csv", width="stretch")

json_data = df.to_json(orient='records')
with ex_col2:
    st.download_button("📜 Download JSON", data=json_data, file_name="department_analytics.json", mime="application/json", width="stretch")

# PDF Generation
def generate_dept_pdf(dept_summary_df):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []
    styles = getSampleStyleSheet()
    
    elements.append(Paragraph("Department Analytics Summary", styles['Title']))
    elements.append(Spacer(1, 12))
    
    data = [dept_summary_df.columns.tolist()] + dept_summary_df.values.tolist()
    
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
    pdf_bytes = generate_dept_pdf(dept_summary.round(2))
    with ex_col3:
        st.download_button("📑 Download Summary PDF", data=pdf_bytes, file_name="Department_Summary.pdf", mime="application/pdf", width="stretch")
except Exception as e:
    with ex_col3:
        st.error("PDF generation failed.")

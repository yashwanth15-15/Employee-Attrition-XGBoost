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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
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
        # Define defaults for numeric optional columns
        numeric_defaults = {
            'prediction_probability': 0.0,
            'monthly_income': 0.0,
            'replacement_cost': 0.0,
            'job_satisfaction': 3.0,
            'environment_satisfaction': 3.0,
            'work_life_balance': 3.0
        }
        for col, default in numeric_defaults.items():
            if col not in df.columns:
                df[col] = default
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)
            
        if 'prediction_date' not in df.columns:
            df['prediction_date'] = pd.Timestamp.now()
        df['prediction_date'] = pd.to_datetime(df['prediction_date'], errors='coerce')
        
        # String/Categorical fallbacks
        for col in ['department', 'gender', 'marital_status', 'overtime', 'business_travel', 'risk_category']:
            if col not in df.columns:
                df[col] = 'Unknown' if col != 'overtime' else 'No'
    return df

raw_df = load_data()

if raw_df.empty:
    st.info("ℹ️ No historical data available. Generate employee predictions to unlock analytics.")
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

if len(df) < 20:
    st.info("ℹ️ **Analytics are based on a limited sample.**\n\nDepartment benchmarking, trends, and predictive insights become significantly more reliable after approximately 20–30 employee predictions.")

# ==========================================
# SECTION 1: EXECUTIVE KPIs
# ==========================================
total_emp = len(df)
high_risk = len(df[df['risk_category'] == 'High'])
high_risk_pct = (high_risk / total_emp) * 100 if total_emp > 0 else 0
med_risk = len(df[df['risk_category'] == 'Medium'])
low_risk = len(df[df['risk_category'] == 'Low'])

avg_risk = df['prediction_probability'].mean() * 100
health_rating = 100 - avg_risk

dep_risk_avg = df.groupby('department')['prediction_probability'].mean().reset_index()
highest_risk_dep = dep_risk_avg.sort_values('prediction_probability', ascending=False).iloc[0]['department'] if not dep_risk_avg.empty else "N/A"
lowest_risk_dep = dep_risk_avg.sort_values('prediction_probability', ascending=True).iloc[0]['department'] if not dep_risk_avg.empty else "N/A"

today = datetime.now().date()
preds_today = len(df[df['prediction_date'].dt.date == today])

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("👥 Total Employees", total_emp)
    st.metric("📈 Highest Risk Dept", highest_risk_dep)
    st.metric("🔥 Attrition Heat Score", f"{avg_risk:.1f} / 100")
with col2:
    st.metric("🔴 High Risk Employees", high_risk)
    st.metric("📉 Lowest Risk Dept", lowest_risk_dep)
    st.metric("🚨 High Risk %", f"{high_risk_pct:.1f}%")
with col3:
    st.metric("🟡 Medium Risk Employees", med_risk)
    st.metric("⚡ Average Attrition Risk", f"{avg_risk:.1f}%")
    st.metric("🏥 Department Health Rating", f"{health_rating:.1f} / 100")
with col4:
    st.metric("🟢 Low Risk Employees", low_risk)
    st.metric("📅 Predictions Today", preds_today)
    
    # Improved Risk Index
    st.metric("📊 Dept Risk Index", f"{int(avg_risk)} / 100")

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

dept_summary['Employee Distribution %'] = (dept_summary['Employees'] / total_emp * 100).round(1)
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
        'Total_Replacement_Cost': 'Replacement Cost (₹)',
        'Employee Distribution %': 'Emp Dist (%)'
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
                                 color_discrete_map=color_map, title='Monthly Income vs Attrition Probability',
                                 hover_data=['department'])
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
st.markdown("#### 🚨 Top Risk Employees")
high_risk_df = df[df['risk_category'] == 'High'].sort_values('prediction_probability', ascending=False)
if not high_risk_df.empty:
    high_risk_table = high_risk_df[['id', 'department', 'prediction_probability', 'replacement_cost']].copy()
    high_risk_table['Risk %'] = (high_risk_table['prediction_probability'] * 100).round(1)
    high_risk_table['Health Score'] = (100 - high_risk_table['Risk %']).astype(int)
    high_risk_table['Rank'] = range(1, len(high_risk_table) + 1)
    high_risk_table['Priority Badge'] = '🚨 Immediate'
    
    # Reorder columns
    high_risk_table = high_risk_table[['Rank', 'id', 'department', 'Risk %', 'Health Score', 'replacement_cost', 'Priority Badge']].rename(columns={
        'id': 'Database ID',
        'department': 'Department',
        'replacement_cost': 'Replacement Cost (₹)'
    })
    
    st.dataframe(high_risk_table.head(10), use_container_width=True, hide_index=True)
else:
    st.success("No high-risk employees found in the current filtered dataset. Excellent retention health!")

st.markdown("---")

# ==========================================
# SECTION 6, 7 & 8: BUSINESS IMPACT & INSIGHTS
# ==========================================
st.subheader("💼 Business Impact & Executive Insights")

b_col1, b_col2 = st.columns([1, 2])

with b_col1:
    # Calculate Business Impact
    high_risk_exposure = high_risk_df['replacement_cost'].sum() if not high_risk_df.empty else 0
    total_workforce_value = df['replacement_cost'].sum()
    highest_cost_dept = dept_summary.sort_values('Total_Replacement_Cost', ascending=False).iloc[0] if not dept_summary.empty else None
    
    with st.container(border=True):
        st.markdown("### Financial Exposure")
        st.metric("High Risk Financial Exposure", f"₹{high_risk_exposure:,.2f}")
        st.metric("Total Workforce Replacement Value", f"₹{total_workforce_value:,.2f}")
        
        if high_risk_exposure == 0:
            st.success("Current Financial Status\n\n🟢 Stable Workforce")
        else:
            if highest_cost_dept is not None:
                st.metric("Highest Cost Department", highest_cost_dept['department'])
                st.metric(f"Exposure in {highest_cost_dept['department']}", f"₹{highest_cost_dept['Total_Replacement_Cost']:,.2f}")

with b_col2:
    with st.container(border=True):
        st.markdown("### 🧠 Auto-Generated Insights")
        
        insights = []
        
        # Single Department Check
        if len(departments) == 1:
            insights.append("Current analytics include one department only. Department-level benchmarking will become available as additional departments are predicted.")
        else:
            if highest_risk_dep != "N/A":
                insights.append(f"**{highest_risk_dep}** currently holds the highest average attrition risk ({dept_summary.iloc[0]['Average_Risk']:.1f}%).")
            if lowest_risk_dep != "N/A":
                insights.append(f"**{lowest_risk_dep}** maintains the strongest retention outlook across the organization.")
            
        # Check overtime correlation
        ot_high = len(df[(df['overtime'] == 'Yes') & (df['risk_category'] == 'High')])
        non_ot_high = len(df[(df['overtime'] == 'No') & (df['risk_category'] == 'High')])
        if ot_high > non_ot_high:
            insights.append("Elevated overtime levels strongly correlate with high attrition profiles in the current population.")
            
        # Check WLB using static terminology
        if not dept_summary.empty and len(departments) > 1:
            lowest_wlb = dept_summary.sort_values('Avg_WLB').iloc[0]
            if lowest_wlb['Avg_WLB'] < 2.5:
                insights.append(f"Average Work-Life Balance in **{lowest_wlb['department']}**: {lowest_wlb['Avg_WLB']:.1f} (Below Organizational Target)")
            
        if len(insights) == 0 or total_emp == 0:
            insights.append("Current workforce metrics are within expected organizational thresholds.")
            
        for insight in insights:
            st.info(f"💡 {insight}")

st.markdown("### 📋 Executive Recommendations")
e_col1, e_col2, e_col3 = st.columns(3)

if high_risk == 0:
    with e_col1:
        with st.container(border=True):
            st.markdown("🔴 **Immediate**")
            st.write("✔ No urgent intervention required.")
    with e_col2:
        with st.container(border=True):
            st.markdown("🟡 **Manager**")
            st.write("✔ Continue periodic employee engagement.")
    with e_col3:
        with st.container(border=True):
            st.markdown("🟢 **Long-Term**")
            st.write("✔ Maintain current retention strategy.")
else:
    with e_col1:
        with st.container(border=True):
            st.markdown("🔴 **Immediate**")
            if highest_risk_dep != "N/A":
                st.write(f"✔ Investigate high risk drivers in {highest_risk_dep}")
            if high_risk_exposure > 0:
                st.write("✔ Review retention strategies for critical profiles")
            else:
                st.write("✔ Review individual critical profiles.")

    with e_col2:
        with st.container(border=True):
            st.markdown("🟡 **Manager**")
            if ot_high > non_ot_high:
                st.write("✔ Reduce overtime workload across departments")
            st.write("✔ Review compensation strategy and workload distribution")

    with e_col3:
        with st.container(border=True):
            st.markdown("🟢 **Long-Term**")
            st.write("✔ Improve promotion and career growth opportunities")
            if lowest_risk_dep != "N/A":
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
def generate_dept_pdf(dept_summary_df, insights_text):
    buffer = BytesIO()
    
    # Custom pagination function
    def add_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(colors.gray)
        canvas.drawString(inch, 0.75 * inch, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        canvas.drawRightString(7.5 * inch, 0.75 * inch, f"Page {doc.page}")
        canvas.restoreState()

    from reportlab.lib.units import inch
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=inch, leftMargin=inch, topMargin=inch, bottomMargin=inch)
    
    elements = []
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    h2_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Cover Section
    elements.append(Paragraph("Enterprise HR Department Analytics Summary", title_style))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Executive insights detailing attrition risk and financial exposure across the organization.", normal_style))
    elements.append(Spacer(1, 24))
    
    # Insights
    elements.append(Paragraph("Executive Insights", h2_style))
    for t in insights_text:
        elements.append(Paragraph(f"• {t}", normal_style))
        elements.append(Spacer(1, 6))
    
    elements.append(Spacer(1, 24))
    
    # Table
    elements.append(Paragraph("Department Overview", h2_style))
    elements.append(Spacer(1, 12))
    
    data = [dept_summary_df.columns.tolist()] + dept_summary_df.values.tolist()
    t = Table(data, style=[
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1f2937')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f9fafb')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#d1d5db'))
    ])
    elements.append(t)
    
    doc.build(elements, onFirstPage=add_footer, onLaterPages=add_footer)
    buffer.seek(0)
    return buffer.read()

try:
    pdf_bytes = generate_dept_pdf(dept_summary.round(2), insights)
    with ex_col3:
        st.download_button("📑 Download Executive PDF", data=pdf_bytes, file_name="Department_Analytics_Report.pdf", mime="application/pdf", width="stretch")
except Exception as e:
    with ex_col3:
        st.error(f"PDF generation failed: {e}")

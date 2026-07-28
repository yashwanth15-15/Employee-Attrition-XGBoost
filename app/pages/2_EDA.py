import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(page_title="Exploratory Data Analysis", layout="wide")

st.title("📊 Exploratory Data Analysis Dashboard")
st.markdown("""
This dashboard provides a comprehensive analysis of the IBM HR Analytics Employee Attrition dataset. 
Use the filters on the left to explore specific segments of the workforce.
""")

# ==========================================
# DATA LOADING & ERROR HANDLING
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("❌ Dataset not found! Please ensure 'data/WA_Fn-UseC_-HR-Employee-Attrition.csv' exists.")
    st.stop()

required_cols = ["Attrition", "Department", "Gender", "MaritalStatus", "BusinessTravel", "OverTime", 
                 "JobSatisfaction", "MonthlyIncome", "Age", "WorkLifeBalance", "YearsAtCompany"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
    st.stop()

# ==========================================
# FILTERS
# ==========================================
st.sidebar.header("🔍 Filters")

def multiselect_filter(label, column):
    options = df[column].dropna().unique().tolist()
    return st.sidebar.multiselect(label, options=options, default=options)

departments = multiselect_filter("Department", "Department")
genders = multiselect_filter("Gender", "Gender")
marital_status = multiselect_filter("Marital Status", "MaritalStatus")
business_travel = multiselect_filter("Business Travel", "BusinessTravel")
overtime = multiselect_filter("OverTime", "OverTime")

filtered_df = df[
    (df["Department"].isin(departments)) &
    (df["Gender"].isin(genders)) &
    (df["MaritalStatus"].isin(marital_status)) &
    (df["BusinessTravel"].isin(business_travel)) &
    (df["OverTime"].isin(overtime))
]

if filtered_df.empty:
    st.warning("⚠️ No data available for the selected filters. Please adjust your selections.")
    st.stop()

# ==========================================
# SUMMARY KPI CARDS
# ==========================================
st.markdown("### 📌 Key Performance Indicators")
total_employees = len(filtered_df)
attrition_count = len(filtered_df[filtered_df["Attrition"] == "Yes"])
no_attrition_count = total_employees - attrition_count
attrition_rate = (attrition_count / total_employees * 100) if total_employees > 0 else 0
avg_income = filtered_df["MonthlyIncome"].mean()
avg_years = filtered_df["YearsAtCompany"].mean()

kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5, kpi_c6 = st.columns(6)
kpi_c1.metric("Total Employees", f"{total_employees:,}")
kpi_c2.metric("Attrition (Yes)", f"{attrition_count:,}")
kpi_c3.metric("Attrition (No)", f"{no_attrition_count:,}")
kpi_c4.metric("Attrition Rate", f"{attrition_rate:.1f}%")
kpi_c5.metric("Avg Monthly Income", f"₹{avg_income:,.0f}")
kpi_c6.metric("Avg Years at Company", f"{avg_years:.1f}")

st.divider()

# Helper for Dynamic Insights
def show_insight(key_insight, business_interpretation, hr_recommendation):
    st.info(f"""
    **🔍 Key Insight:** {key_insight}  
    **🏢 Business Interpretation:** {business_interpretation}  
    **💡 HR Recommendation:** {hr_recommendation}
    """)

# ==========================================
# 1. ATTRITION DISTRIBUTION
# ==========================================
st.subheader("1. Attrition Distribution")
attr_counts = filtered_df["Attrition"].value_counts().reset_index()
attr_counts.columns = ["Attrition", "Count"]
fig1 = px.pie(attr_counts, names="Attrition", values="Count", title="Overall Attrition Distribution", hole=0.4)
st.plotly_chart(fig1, use_container_width=True)

show_insight(
    key_insight=f"The current attrition rate is {attrition_rate:.1f}%.",
    business_interpretation="A rate above 15% generally indicates higher than average turnover, leading to increased hiring and training costs.",
    hr_recommendation="Monitor high-risk groups and focus on retention initiatives if the rate exceeds industry benchmarks."
)

st.divider()

# ==========================================
# 2. DEPARTMENT-WISE ATTRITION
# ==========================================
st.subheader("2. Department-wise Attrition")
dept_attr = filtered_df.groupby(["Department", "Attrition"]).size().reset_index(name="Count")
fig2 = px.bar(dept_attr, x="Department", y="Count", color="Attrition", barmode="group", title="Attrition by Department")
st.plotly_chart(fig2, use_container_width=True)

# Compute Insight
if not filtered_df[filtered_df["Attrition"] == "Yes"].empty:
    dept_rates = filtered_df.groupby("Department").apply(lambda x: (x["Attrition"] == "Yes").mean()).sort_values(ascending=False)
    highest_dept = dept_rates.index[0]
    highest_rate = dept_rates.iloc[0] * 100
else:
    highest_dept = "N/A"
    highest_rate = 0

show_insight(
    key_insight=f"{highest_dept} department has the highest attrition rate at {highest_rate:.1f}%.",
    business_interpretation="Specific departments may experience different stressors, management styles, or lack of growth opportunities.",
    hr_recommendation=f"Conduct stay interviews within the {highest_dept} department to uncover specific pain points."
)

st.divider()

# ==========================================
# 3. GENDER VS ATTRITION
# ==========================================
st.subheader("3. Gender vs Attrition")
gender_attr = filtered_df.groupby(["Gender", "Attrition"]).size().reset_index(name="Count")
fig3 = px.bar(gender_attr, x="Gender", y="Count", color="Attrition", barmode="group", title="Attrition by Gender")
st.plotly_chart(fig3, use_container_width=True)

if not filtered_df[filtered_df["Attrition"] == "Yes"].empty:
    gender_rates = filtered_df.groupby("Gender").apply(lambda x: (x["Attrition"] == "Yes").mean()).sort_values(ascending=False)
    highest_gender = gender_rates.index[0]
    highest_grate = gender_rates.iloc[0] * 100
else:
    highest_gender = "N/A"
    highest_grate = 0

show_insight(
    key_insight=f"{highest_gender} employees exhibit a {highest_grate:.1f}% attrition rate.",
    business_interpretation="Significant gender disparities in attrition can indicate inclusion issues, unequal pay, or lack of support systems.",
    hr_recommendation="Review diversity and inclusion policies, and ensure equal compensation and career progression."
)

st.divider()

# ==========================================
# 4. OVERTIME VS ATTRITION
# ==========================================
st.subheader("4. OverTime vs Attrition")
ot_attr = filtered_df.groupby(["OverTime", "Attrition"]).size().reset_index(name="Count")
fig4 = px.bar(ot_attr, x="OverTime", y="Count", color="Attrition", barmode="group", title="Impact of OverTime on Attrition")
st.plotly_chart(fig4, use_container_width=True)

if not filtered_df[filtered_df["Attrition"] == "Yes"].empty:
    ot_rates = filtered_df.groupby("OverTime").apply(lambda x: (x["Attrition"] == "Yes").mean())
    ot_yes_rate = ot_rates.get("Yes", 0) * 100
    ot_no_rate = ot_rates.get("No", 0) * 100
else:
    ot_yes_rate, ot_no_rate = 0, 0

show_insight(
    key_insight=f"Employees working overtime have an attrition rate of {ot_yes_rate:.1f}% compared to {ot_no_rate:.1f}% for those who do not.",
    business_interpretation="Consistent overtime often leads to burnout, reduced productivity, and ultimately resignation.",
    hr_recommendation="Implement strict overtime caps, hire additional staff, or redistribute workload to prevent burnout."
)

st.divider()

# ==========================================
# 5. JOB SATISFACTION VS ATTRITION
# ==========================================
st.subheader("5. Job Satisfaction vs Attrition")
fig5 = px.box(filtered_df, x="Attrition", y="JobSatisfaction", color="Attrition", title="Job Satisfaction Levels by Attrition")
st.plotly_chart(fig5, use_container_width=True)

js_left = filtered_df[filtered_df["Attrition"] == "Yes"]["JobSatisfaction"].mean() if not filtered_df[filtered_df["Attrition"] == "Yes"].empty else 0
js_stay = filtered_df[filtered_df["Attrition"] == "No"]["JobSatisfaction"].mean() if not filtered_df[filtered_df["Attrition"] == "No"].empty else 0

show_insight(
    key_insight=f"Average Job Satisfaction is {js_left:.1f}/4 for departing employees vs {js_stay:.1f}/4 for remaining employees.",
    business_interpretation="Lower satisfaction is a leading indicator of turnover, often driven by poor management or unfulfilling work.",
    hr_recommendation="Launch anonymous employee pulse surveys and establish continuous feedback loops with management."
)

st.divider()

# ==========================================
# 6. MONTHLY INCOME DISTRIBUTION
# ==========================================
st.subheader("6. Monthly Income Distribution")
fig6 = px.histogram(filtered_df, x="MonthlyIncome", color="Attrition", marginal="box", nbins=30, title="Monthly Income Spread and Attrition")
st.plotly_chart(fig6, use_container_width=True)

income_left = filtered_df[filtered_df["Attrition"] == "Yes"]["MonthlyIncome"].mean() if not filtered_df[filtered_df["Attrition"] == "Yes"].empty else 0
income_stay = filtered_df[filtered_df["Attrition"] == "No"]["MonthlyIncome"].mean() if not filtered_df[filtered_df["Attrition"] == "No"].empty else 0

show_insight(
    key_insight=f"Departing employees earn an average of ₹{income_left:,.0f}/mo vs ₹{income_stay:,.0f}/mo for those who stay.",
    business_interpretation="Compensation below market rate or internal peer levels is a primary driver for seeking external opportunities.",
    hr_recommendation="Conduct a comprehensive market salary review and consider equity adjustments for critical roles."
)

st.divider()

# ==========================================
# 7. AGE DISTRIBUTION
# ==========================================
st.subheader("7. Age Distribution")
fig7 = px.histogram(filtered_df, x="Age", color="Attrition", marginal="box", nbins=30, title="Age Demographics and Attrition")
st.plotly_chart(fig7, use_container_width=True)

age_left = filtered_df[filtered_df["Attrition"] == "Yes"]["Age"].mean() if not filtered_df[filtered_df["Attrition"] == "Yes"].empty else 0
age_stay = filtered_df[filtered_df["Attrition"] == "No"]["Age"].mean() if not filtered_df[filtered_df["Attrition"] == "No"].empty else 0

show_insight(
    key_insight=f"The average age of resigning employees is {age_left:.1f} years, compared to {age_stay:.1f} years for retained staff.",
    business_interpretation="Younger employees may seek rapid career progression and frequent job changes, while older employees value stability.",
    hr_recommendation="Tailor retention strategies: offer mentorship and fast-track career paths for younger demographics."
)

st.divider()

# ==========================================
# 8. WORK LIFE BALANCE VS ATTRITION
# ==========================================
st.subheader("8. Work Life Balance vs Attrition")
wlb_attr = filtered_df.groupby(["WorkLifeBalance", "Attrition"]).size().reset_index(name="Count")
fig8 = px.bar(wlb_attr, x="WorkLifeBalance", y="Count", color="Attrition", barmode="group", title="Work Life Balance Rating (1=Bad to 4=Excellent)")
st.plotly_chart(fig8, use_container_width=True)

if not filtered_df[filtered_df["Attrition"] == "Yes"].empty:
    wlb_rates = filtered_df.groupby("WorkLifeBalance").apply(lambda x: (x["Attrition"] == "Yes").mean())
    wlb_lowest = wlb_rates.get(1, 0) * 100
else:
    wlb_lowest = 0

show_insight(
    key_insight=f"Employees rating their Work Life Balance as '1' (Bad) have a {wlb_lowest:.1f}% attrition rate.",
    business_interpretation="A poor work-life balance disrupts personal well-being and is unsustainable long-term.",
    hr_recommendation="Introduce flexible working hours, remote work options, or wellness days to improve balance."
)

st.divider()

# ==========================================
# 9 & 10. CORRELATION HEATMAP & TOP FEATURES
# ==========================================
st.subheader("9. Correlation Heatmap & Top Predictors")

# Prepare numeric data for correlation
corr_df = filtered_df.copy()
# Map attrition to numeric
corr_df["Attrition_Num"] = corr_df["Attrition"].apply(lambda x: 1 if x == "Yes" else 0)
numeric_df = corr_df.select_dtypes(include=['number'])

if numeric_df.shape[1] > 1:
    corr_matrix = numeric_df.corr()
    
    # 9. Heatmap
    fig9 = px.imshow(corr_matrix, text_auto=False, aspect="auto", color_continuous_scale="RdBu_r", title="Feature Correlation Heatmap")
    st.plotly_chart(fig9, use_container_width=True)

    # 10. Top Correlated Features
    st.markdown("#### 10. Top Features Correlated with Attrition")
    attr_corr = corr_matrix["Attrition_Num"].drop("Attrition_Num").sort_values(key=abs, ascending=False).reset_index()
    attr_corr.columns = ["Feature", "Correlation Coefficient"]
    st.dataframe(attr_corr.style.background_gradient(cmap="RdBu_r"), use_container_width=True)
    
    if not attr_corr.empty:
        top_feature = attr_corr.iloc[0]["Feature"]
        top_corr_val = attr_corr.iloc[0]["Correlation Coefficient"]
        direction = "positively" if top_corr_val > 0 else "negatively"
        
        show_insight(
            key_insight=f"'{top_feature}' is the most strongly correlated feature with Attrition ({top_corr_val:.2f}).",
            business_interpretation=f"As '{top_feature}' increases, attrition probability {direction} scales, indicating a direct systemic relationship.",
            hr_recommendation=f"Prioritize initiatives that directly optimize '{top_feature}' to maximize retention ROI."
        )
else:
    st.info("Not enough numeric data to compute correlations.")

st.divider()

# ==========================================
# EXPORT FUNCTIONALITY
# ==========================================
st.subheader("📥 Export Filtered Dataset")
col_csv, col_excel = st.columns(2)

with col_csv:
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data (CSV)",
        data=csv_data,
        file_name="filtered_eda_data.csv",
        mime="text/csv"
    )

with col_excel:
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        filtered_df.to_excel(writer, index=False, sheet_name='Filtered Data')
    st.download_button(
        label="Download Filtered Data (Excel)",
        data=excel_buffer.getvalue(),
        file_name="filtered_eda_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.caption("Generated by Employee Attrition Prediction System | Dynamic EDA Module")
from datetime import datetime
from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from helpers.executive_dashboard import (
    compute_insights,
    compute_kpis,
    generate_pdf_summary,
    load_data,
)

st.set_page_config(page_title="Executive Dashboard", layout="wide")

st.title("📊 Executive HR Analytics Dashboard")

# ---------- Sidebar Filters ----------
st.sidebar.header("Filters")
# Department filter (multi-select) – will be populated after data load
# Risk Category filter
risk_options = ["High", "Medium", "Low"]
selected_risks = st.sidebar.multiselect(
    "Risk Category", options=risk_options, default=risk_options
)
# Date range filter
start_date = st.sidebar.date_input("Start Date", value=None)
end_date = st.sidebar.date_input("End Date", value=None)
# Health Score range
min_health, max_health = st.sidebar.slider(
    "Health Score", min_value=0, max_value=100, value=(0, 100)
)
# Monthly Income range
min_income, max_income = st.sidebar.slider(
    "Monthly Income", min_value=0, max_value=1000000, step=5000, value=(0, 1000000)
)

# ---------- Load & Filter Data ----------
raw_df = load_data()
if raw_df.empty:
    st.info("No prediction records available.")
    st.stop()

# Populate department filter options dynamically
departments = sorted(raw_df["department"].dropna().unique())
selected_departments = st.sidebar.multiselect(
    "Department", options=departments, default=departments
)

# Build filter dict for helper
filters = {
    "department": selected_departments,
    "risk_category": selected_risks,
    "start_date": start_date.strftime("%Y-%m-%d") if start_date else None,
    "end_date": end_date.strftime("%Y-%m-%d") if end_date else None,
    "min_health": min_health if min_health != 0 else None,
    "max_health": max_health if max_health != 100 else None,
    "min_income": min_income if min_income != 0 else None,
    "max_income": max_income if max_income != 1000000 else None,
}

with st.spinner("Loading Executive Dashboard Data..."):
    df = load_data(filters)
    kpis = compute_kpis(df)
    insights = compute_insights(df)

# ---------- KPI Cards ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Employees Predicted", kpis["total"])
col2.metric("High Risk Employees", kpis["high"])
col3.metric("Medium Risk Employees", kpis["medium"])
col4.metric("Low Risk Employees", kpis["low"])
col5, col6, col7, col8 = st.columns(4)
col5.metric("Avg Attrition Prob.", f"{kpis['avg_prob']*100:.2f}%")
col6.metric("Avg Health Score", f"{kpis['avg_health']:.1f}")
col7.metric("Avg Monthly Income", f"₹{kpis['avg_income']:,.0f}")
col8.metric("Est. Total Replacement Cost", f"₹{kpis['total_cost']:,.0f}")

st.divider()

# ---------- Advanced Insights ----------
with st.expander("🔍 Automated Insights"):
    for insight in insights:
        st.markdown(insight, unsafe_allow_html=True)

st.divider()

# ---------- Charts ----------
# 1. Department-wise Attrition Rate (Bar)
dept_counts = df.groupby("department").size().reset_index(name="count")
dept_prob = (
    df.groupby("department")["prediction_probability"]
    .mean()
    .reset_index(name="avg_prob")
)
dept_chart = px.bar(
    dept_prob,
    x="department",
    y="avg_prob",
    title="Dept‑wise Avg Attrition Probability",
    labels={"avg_prob": "Avg Probability"},
)
st.plotly_chart(dept_chart, width="stretch")

# 2. Risk Category Distribution (Donut)
risk_counts = df["risk_category"].value_counts().reset_index()
risk_counts.columns = ["risk_category", "count"]
fig_risk = go.Figure(
    data=[
        go.Pie(
            labels=risk_counts["risk_category"], values=risk_counts["count"], hole=0.4
        )
    ]
)
fig_risk.update_layout(title_text="Risk Category Distribution")
st.plotly_chart(fig_risk, width="stretch")

# 3. Prediction Trend Over Time (Line)
trend_df = df.copy()
trend_df["date"] = pd.to_datetime(trend_df["prediction_date"]).dt.date
trend = trend_df.groupby("date")["prediction_probability"].mean().reset_index()

if len(df) < 20:
    st.info("Trend analysis becomes available after at least 20 employee predictions.")
else:
    fig_trend = px.line(
        trend,
        x="date",
        y="prediction_probability",
        title="Avg Attrition Probability Over Time",
        labels={"prediction_probability": "Avg Probability"},
    )
    st.plotly_chart(fig_trend, width="stretch")

# 4. Health Score Distribution (Histogram)
fig_health = px.histogram(
    df, x="health_score", nbins=20, title="Health Score Distribution"
)
st.plotly_chart(fig_health, width="stretch")

# 5. Monthly Income vs Attrition Probability (Scatter)
fig_scatter = px.scatter(
    df,
    x="monthly_income",
    y="prediction_probability",
    color="risk_category",
    title="Income vs Attrition Probability",
    labels={
        "monthly_income": "Monthly Income",
        "prediction_probability": "Attrition Prob.",
    },
)
st.plotly_chart(fig_scatter, width="stretch")

# 6. Department Average Health Score (Horizontal Bar)
dept_health = df.groupby("department")["health_score"].mean().reset_index()
fig_dept_health = px.bar(
    dept_health,
    y="department",
    x="health_score",
    orientation="h",
    title="Dept Avg Health Score",
)
st.plotly_chart(fig_dept_health, width="stretch")

# 7. Replacement Cost by Department (Treemap)
dept_cost = df.groupby("department")["replacement_cost"].sum().reset_index()
fig_treemap = px.treemap(
    dept_cost,
    path=["department"],
    values="replacement_cost",
    title="Replacement Cost by Department",
)
st.plotly_chart(fig_treemap, width="stretch")

# 8. Overtime vs Risk Category (Stacked Bar)
overtime_risk = (
    df.groupby(["overtime", "risk_category"]).size().reset_index(name="count")
)
fig_overtime = px.bar(
    overtime_risk,
    x="overtime",
    y="count",
    color="risk_category",
    title="Overtime vs Risk Category",
    barmode="stack",
)
st.plotly_chart(fig_overtime, width="stretch")

st.divider()

# ---------- Top Risk Table ----------
st.subheader("Top 10 Highest Risk Employees")
if not df.empty:
    top10 = df.nlargest(10, "prediction_probability")[
        [
            "employee_name",
            "age",
            "gender",
            "department",
            "prediction_probability",
            "risk_category",
            "health_score",
            "monthly_income",
            "replacement_cost",
        ]
    ]
    top10 = top10.rename(
        columns={
            "prediction_probability": "Prob",
            "monthly_income": "Income",
            "replacement_cost": "Cost",
        }
    )
    st.dataframe(top10, width="stretch", hide_index=True)

st.divider()

# ---------- Export Section ----------
st.subheader("Export Data")
col_csv, col_excel, col_pdf = st.columns(3)
with col_csv:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export CSV",
        data=csv_bytes,
        file_name=f"employee_attrition_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
with col_excel:
    # Pandas Excel writer uses openpyxl engine
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Predictions")
    st.download_button(
        label="📥 Export Excel",
        data=excel_buffer.getvalue(),
        file_name=f"employee_attrition_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
with col_pdf:
    pdf_buffer = generate_pdf_summary(kpis, insights)
    st.download_button(
        label="📥 Export PDF Summary",
        data=pdf_buffer,
        file_name=f"employee_attrition_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
    )

st.caption("Generated by Employee Attrition Prediction System – Executive Dashboard")

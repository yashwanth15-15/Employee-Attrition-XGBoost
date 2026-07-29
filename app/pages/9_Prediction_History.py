from datetime import datetime
from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st
from database import (delete_all_predictions, delete_predictions,
                      get_predictions)

st.set_page_config(page_title="Prediction History", layout="wide")

st.title("📜 Prediction History")

# Fetch all predictions first to populate filters dynamically
all_preds = get_predictions({})
available_departments = (
    all_preds["department"].unique().tolist()
    if not all_preds.empty and "department" in all_preds.columns
    else []
)

# Sidebar Filters
st.sidebar.header("Filters")
# Department filter (multiselect)
departments = st.sidebar.multiselect(
    "Department", options=available_departments, default=[]
)
# Risk Category filter
risk_categories = st.sidebar.multiselect(
    "Risk Category", options=["High", "Medium", "Low"], default=[]
)
# Date range
start_date = st.sidebar.date_input("Start Date", value=None)
end_date = st.sidebar.date_input("End Date", value=None)
# Age range
min_age, max_age = st.sidebar.slider(
    "Age Range", min_value=0, max_value=100, value=(0, 100)
)

# Build filter dict
filters = {}
if departments:
    filters["department"] = departments
if risk_categories:
    filters["risk_category"] = risk_categories
if start_date:
    filters["start_date"] = start_date.strftime("%Y-%m-%d")
if end_date:
    filters["end_date"] = end_date.strftime("%Y-%m-%d")
if min_age != 0:
    filters["min_age"] = min_age
if max_age != 100:
    filters["max_age"] = max_age

# Load data
df = get_predictions(filters)

if df.empty:
    st.info("No prediction records found.")
    st.stop()

# Ensure minimal columns to display history
if "id" not in df.columns:
    st.warning("The database is missing the 'id' column. Cannot display history.")
    st.stop()

# KPI Cards
col_total, col_high, col_med, col_low = st.columns(4)
col_total.metric("Total Predictions", f"{len(df)}")
if "risk_category" in df.columns:
    col_high.metric("High Risk", f"{df['risk_category'].str.lower().eq('high').sum()}")
    col_med.metric(
        "Medium Risk", f"{df['risk_category'].str.lower().eq('medium').sum()}"
    )
    col_low.metric("Low Risk", f"{df['risk_category'].str.lower().eq('low').sum()}")
else:
    col_high.metric("High Risk", "N/A")
    col_med.metric("Medium Risk", "N/A")
    col_low.metric("Low Risk", "N/A")

# Export Buttons
export_col1, export_col2 = st.columns(2)
with export_col1:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Export CSV",
        data=csv,
        file_name=f"employee_attrition_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
with export_col2:
    excel_buffer = BytesIO()
    df.to_excel(excel_buffer, index=False, engine="openpyxl")
    st.download_button(
        "📥 Export Excel",
        data=excel_buffer.getvalue(),
        file_name=f"employee_attrition_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# Delete Section
st.subheader("Delete Predictions")
selected_ids = st.multiselect(
    "Select rows to delete (by ID)", options=df["id"].tolist()
)
if st.button("Delete Selected"):
    if selected_ids:
        confirm = st.checkbox("Confirm deletion of selected rows")
        if confirm:
            delete_predictions(selected_ids)
            st.success(f"Deleted {len(selected_ids)} records.")
            st.rerun()
        else:
            st.warning("Please confirm deletion.")
    else:
        st.info("No rows selected.")

if st.button("Delete All Predictions"):
    confirm_all = st.checkbox("Confirm delete ALL predictions")
    if confirm_all:
        delete_all_predictions()
        st.success("All predictions deleted.")
        st.rerun()
    else:
        st.warning("Please confirm deletion of all records.")

# Data Table with pagination
page_size = 20
total_pages = (len(df) - 1) // page_size + 1
page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
start_idx = (page - 1) * page_size
end_idx = start_idx + page_size
paged_df = df.iloc[start_idx:end_idx]
st.dataframe(paged_df, hide_index=True, width="stretch")

# Visualizations
st.subheader("Visualizations")

# Department-wise predictions bar chart
if "department" in df.columns:
    dept_counts = df["department"].value_counts().reset_index()
    dept_counts.columns = ["department", "count"]
    fig_dept = px.bar(
        dept_counts, x="department", y="count", title="Predictions per Department"
    )
    st.plotly_chart(fig_dept, width="stretch")
else:
    st.warning("Predictions per Department chart skipped (department missing).")

# Risk category pie chart
if "risk_category" in df.columns:
    risk_counts = df["risk_category"].value_counts().reset_index()
    risk_counts.columns = ["risk_category", "count"]
    fig_risk = px.pie(
        risk_counts,
        names="risk_category",
        values="count",
        title="Risk Category Distribution",
    )
    st.plotly_chart(fig_risk, width="stretch")
else:
    st.warning("Risk Category Distribution chart skipped (risk_category missing).")

# Prediction trend over time line chart
if "prediction_date" in df.columns and "prediction_probability" in df.columns:
    df["prediction_date"] = pd.to_datetime(df["prediction_date"])
    trend = (
        df.groupby(df["prediction_date"].dt.date)["prediction_probability"]
        .mean()
        .reset_index()
    )
    trend.columns = ["date", "avg_probability"]

    if len(df) < 20:
        st.info(
            "Trend analysis becomes available after at least 20 employee predictions."
        )
    else:
        fig_trend = px.line(
            trend,
            x="date",
            y="avg_probability",
            title="Average Prediction Probability Over Time",
        )
        st.plotly_chart(fig_trend, width="stretch")
else:
    st.warning(
        "Prediction trend chart skipped (prediction_date or prediction_probability missing)."
    )

# Health Score distribution histogram
if "health_score" in df.columns:
    fig_health = px.histogram(
        df, x="health_score", nbins=20, title="Health Score Distribution"
    )
    st.plotly_chart(fig_health, width="stretch")
else:
    st.warning("Health Score Distribution chart skipped (health_score missing).")

st.caption(
    "Generated by Employee Attrition Prediction System | Prediction History Module"
)

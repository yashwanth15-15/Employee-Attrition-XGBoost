import os

# Import database function
import sys
from datetime import datetime
from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_predictions
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# For PDF export
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

st.set_page_config(page_title="Retention Cost Planning", layout="wide", page_icon="💰")

# ==========================================
# HEADER
# ==========================================
st.title("💰 Retention Cost & Workforce Planning")
st.markdown(
    "Executive financial dashboard estimating organizational attrition cost and retention program ROI."
)


# ==========================================
# FETCH DATA
# ==========================================
@st.cache_data(ttl=60)
def load_data():
    df = get_predictions()
    if not df.empty:
        # Define defaults for numeric optional columns
        numeric_defaults = {
            "prediction_probability": 0.0,
            "monthly_income": 0.0,
            "replacement_cost": 0.0,
            "job_satisfaction": 3.0,
            "work_life_balance": 3.0,
        }
        for col, default in numeric_defaults.items():
            if col not in df.columns:
                df[col] = default
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

        if "prediction_date" not in df.columns:
            df["prediction_date"] = pd.Timestamp.now()
        df["prediction_date"] = pd.to_datetime(df["prediction_date"], errors="coerce")

        # String/Categorical fallbacks
        for col in ["department", "risk_category", "overtime"]:
            if col not in df.columns:
                df[col] = "Unknown" if col != "overtime" else "No"
    return df


df = load_data()

if df.empty:
    st.info(
        "ℹ️ No historical data available. Generate employee predictions to unlock financial analytics."
    )
    st.stop()

# ==========================================
# CORE FINANCIAL CALCULATIONS
# ==========================================
total_workforce = len(df)
high_risk_df = df[df["risk_category"] == "High"]
high_risk_count = len(high_risk_df)

total_replacement_cost = df["replacement_cost"].sum()
annual_attrition_cost = high_risk_df["replacement_cost"].sum()
avg_replacement_cost = df["replacement_cost"].mean() if total_workforce > 0 else 0

dept_costs = df.groupby("department")["replacement_cost"].sum().reset_index()
highest_cost_dept = (
    dept_costs.sort_values("replacement_cost", ascending=False).iloc[0]["department"]
    if not dept_costs.empty
    else "N/A"
)

health_score = 100 - (
    len(high_risk_df) / total_workforce * 100 if total_workforce > 0 else 0
)

# ==========================================
# SECTION 1: EXECUTIVE KPIs
# ==========================================
st.markdown("### 🏦 Executive Financial Summary")

k1, k2, k3, k4 = st.columns(4)
k1.metric("👥 Total Workforce", total_workforce)
k2.metric("🔴 High Risk Employees", high_risk_count)
k3.metric("💸 Total Org Replacement Cost", f"₹{total_replacement_cost:,.2f}")
k4.metric("🚨 Current Exposure", f"₹{annual_attrition_cost:,.2f}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("📊 Average Replacement Cost", f"₹{avg_replacement_cost:,.2f}")
c2.metric("🏢 Highest Cost Dept", highest_cost_dept)
c3.metric("🎯 Retention Readiness Score", f"{int(health_score)}/100")
c4.metric(
    "📈 Financial Health",
    (
        "Optimal"
        if high_risk_count == 0
        else (
            "Needs Attention" if high_risk_count > (total_workforce * 0.1) else "Stable"
        )
    ),
)

st.markdown("---")

# Handle Zero Risk State Gracefully
if high_risk_count == 0:
    st.success(
        "🟢 **Excellent Workforce Stability**\n\nNo immediate financial exposure from high-risk attrition has been detected. Retention budget can remain unallocated."
    )

    # We set default variables so code below doesn't break, but hide the what-if visual
    success_rate = 15
    retention_budget = 0
    savings = 0
    roi = 0
else:
    # ==========================================
    # SECTION 3: WHAT-IF BUSINESS SCENARIOS
    # ==========================================
    st.subheader("🧪 What-If Business Scenarios")
    st.write(
        "Simulate the financial impact of running a retention program that successfully reduces high-risk attrition."
    )

    sim_col1, sim_col2 = st.columns([1, 2])

    with sim_col1:
        with st.container(border=True):
            st.markdown("#### Program Success Rate")
            success_rate = st.slider(
                "Target Attrition Reduction (%)",
                min_value=0,
                max_value=50,
                step=5,
                value=15,
            )

            retention_budget = annual_attrition_cost * 0.10
            st.caption(
                f"Assuming a fixed budget of **₹{retention_budget:,.2f}** (10% of total risk exposure)."
            )

    with sim_col2:
        savings = annual_attrition_cost * (success_rate / 100)
        remaining_cost = annual_attrition_cost - savings
        roi = (
            ((savings - retention_budget) / retention_budget) * 100
            if retention_budget > 0
            else 0
        )

        with st.container(border=True):
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Potential Savings", f"₹{savings:,.2f}")
            sc2.metric("Remaining Financial Exposure", f"₹{remaining_cost:,.2f}")

            if roi > 0:
                sc3.metric("Estimated ROI", f"+{roi:,.1f}%")
            elif roi == 0:
                sc3.metric("Estimated ROI", "0.0%")
            else:
                sc3.metric("Estimated ROI", f"{roi:,.1f}%", delta_color="inverse")

    st.markdown("---")

# ==========================================
# SECTION 2: FINANCIAL ANALYTICS
# ==========================================
if high_risk_count > 0:
    st.subheader("📊 Financial Analytics")

    v1, v2 = st.columns(2)
    with v1:
        # Replacement Cost by Department
        fig_bar = px.bar(
            dept_costs,
            x="department",
            y="replacement_cost",
            color="replacement_cost",
            color_continuous_scale="Reds",
            title="Total Financial Exposure by Department",
        )
        fig_bar.update_layout(
            yaxis_title="Replacement Cost (₹)", xaxis_title="Department"
        )
        st.plotly_chart(fig_bar, width="stretch")

    with v2:
        # Cost vs Risk Bubble Chart
        fig_bubble = px.scatter(
            df,
            x="prediction_probability",
            y="replacement_cost",
            size="monthly_income",
            color="risk_category",
            hover_name="department",
            color_discrete_map={
                "High": "salmon",
                "Medium": "gold",
                "Low": "lightgreen",
            },
            title="Cost vs Risk (Bubble Size = Income)",
        )
        fig_bubble.update_layout(
            xaxis_title="Attrition Risk Probability", yaxis_title="Replacement Cost (₹)"
        )
        st.plotly_chart(fig_bubble, width="stretch")

    v3, v4 = st.columns(2)
    with v3:
        # Potential Savings Gauge
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=savings,
                title={"text": f"Potential Savings at {success_rate}% Success (₹)"},
                gauge={
                    "axis": {"range": [0, annual_attrition_cost]},
                    "bar": {"color": "green"},
                    "steps": [
                        {
                            "range": [0, annual_attrition_cost * 0.3],
                            "color": "lightgray",
                        },
                        {
                            "range": [
                                annual_attrition_cost * 0.3,
                                annual_attrition_cost * 0.7,
                            ],
                            "color": "silver",
                        },
                    ],
                },
            )
        )
        fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, width="stretch")

    with v4:
        # Budget Allocation Pie Chart
        budget_dist = pd.DataFrame(
            {
                "Category": ["Rewards", "Training", "Recruiting", "Engagement"],
                "Amount": [
                    retention_budget * 0.4,
                    retention_budget * 0.3,
                    retention_budget * 0.2,
                    retention_budget * 0.1,
                ],
            }
        )

        if retention_budget > 0:
            fig_budget_pie = px.pie(
                budget_dist,
                names="Category",
                values="Amount",
                hole=0.5,
                title="Recommended Budget Allocation",
                color_discrete_sequence=px.colors.sequential.Blues_r,
            )
            st.plotly_chart(fig_budget_pie, width="stretch")

    st.markdown("---")

# ==========================================
# SECTION 5: PRIORITY MATRIX
# ==========================================
st.subheader("🏆 Retention Priority Matrix")

dept_metrics = (
    df.groupby("department")
    .agg(
        Employee_Count=("id", "count"),
        Financial_Risk=(
            "replacement_cost",
            lambda x: x[df.loc[x.index, "risk_category"] == "High"].sum(),
        ),
        High_Risk_Count=("risk_category", lambda x: (x == "High").sum()),
        Avg_Risk=("prediction_probability", "mean"),
    )
    .reset_index()
)

dept_metrics["Retention_Priority"] = (
    dept_metrics["Financial_Risk"].rank(ascending=False, method="min").astype(int)
)
dept_metrics = dept_metrics.sort_values("Retention_Priority")

st.dataframe(
    dept_metrics.rename(
        columns={
            "department": "Department",
            "Employee_Count": "Total Headcount",
            "Financial_Risk": "Financial Risk (₹)",
            "High_Risk_Count": "High Risk Count",
            "Avg_Risk": "Avg Risk Probability",
            "Retention_Priority": "Priority Rank",
        }
    ).style.format(
        {"Financial Risk (₹)": "₹{:,.2f}", "Avg Risk Probability": "{:.2%}"}
    ),
    width="stretch",
    hide_index=True,
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
        if high_risk_count == 0:
            insights.append(
                "Current workforce demonstrates excellent retention health."
            )
            insights.append(
                "No departments currently require urgent retention investment."
            )
            insights.append(
                "Continue periodic monitoring and employee engagement initiatives."
            )
        else:
            if total_replacement_cost > 0:
                high_risk_pct = (annual_attrition_cost / total_replacement_cost) * 100
                insights.append(
                    f"High-risk employees account for **{high_risk_pct:.1f}%** of the total organizational replacement value."
                )

            if not dept_metrics.empty:
                top_dept = dept_metrics.iloc[0]
                if top_dept["Financial_Risk"] > 0:
                    insights.append(
                        f"**{top_dept['department']}** is the highest priority department, contributing **₹{top_dept['Financial_Risk']:,.2f}** to projected attrition costs."
                    )

            ot_high = df[(df["overtime"] == "Yes") & (df["risk_category"] == "High")]
            if not ot_high.empty:
                ot_cost = ot_high["replacement_cost"].sum()
                insights.append(
                    f"Reducing overtime for high-risk profiles could intercept up to **₹{ot_cost:,.2f}** in attrition exposure."
                )

            insights.append(
                f"An intervention achieving a {success_rate}% success rate will save approximately **₹{savings:,.2f}** annually."
            )

        for insight in insights:
            st.info(f"💡 {insight}")

with b_col2:
    with st.container(border=True):
        st.markdown("#### 💰 Suggested Budget Allocation")

        if high_risk_count == 0:
            st.info(
                "No retention budget allocation is currently required because no high-risk workforce exposure exists."
            )
        else:
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

csv_data = dept_metrics.to_csv(index=False).encode("utf-8")
with ex_col1:
    st.download_button(
        "📄 Download Priority CSV",
        data=csv_data,
        file_name=f"employee_attrition_retention_priority_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        width="stretch",
    )

json_data = dept_metrics.to_json(orient="records")
with ex_col2:
    st.download_button(
        "📜 Download Priority JSON",
        data=json_data,
        file_name=f"employee_attrition_retention_priority_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        width="stretch",
    )


# PDF Generation
def generate_financial_pdf(dept_metrics_df, total_cost, target_savings, insights_text):
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

    elements.append(Paragraph("Enterprise Financial Retention Report", title_style))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Executive Summary", h2_style))
    elements.append(
        Paragraph(
            f"<b>Total Potential Attrition Exposure:</b> INR {total_cost:,.2f}",
            normal_style,
        )
    )
    elements.append(
        Paragraph(f"<b>Target Savings:</b> INR {target_savings:,.2f}", normal_style)
    )
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Insights", h2_style))
    for ins in insights_text:
        elements.append(Paragraph(f"• {ins}", normal_style))
        elements.append(Spacer(1, 6))

    elements.append(Spacer(1, 12))

    data = [dept_metrics_df.columns.tolist()] + dept_metrics_df.round(2).values.tolist()

    t = Table(data)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#bdc3c7")),
            ]
        )
    )
    elements.append(t)

    doc.build(elements, onFirstPage=add_footer, onLaterPages=add_footer)
    buffer.seek(0)
    return buffer.read()


try:
    pdf_bytes = generate_financial_pdf(
        dept_metrics, annual_attrition_cost, savings, insights
    )
    with ex_col3:
        st.download_button(
            "📑 Download Financial PDF",
            data=pdf_bytes,
            file_name=f"employee_attrition_financial_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            width="stretch",
        )
except Exception as e:
    with ex_col3:
        st.error("PDF generation failed.")

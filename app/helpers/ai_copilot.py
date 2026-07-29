import re
from datetime import datetime
from io import BytesIO

import pandas as pd
import plotly.express as px
from database import get_predictions
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# ---------------------------------------------------
# Data loading helper
# ---------------------------------------------------
def load_data(filters=None) -> pd.DataFrame:
    """Fetch prediction history from SQLite and apply optional filters.
    ``filters`` may contain keys: department (list), risk_category (list),
    start_date, end_date (YYYY-MM-DD strings), min_health, max_health,
    min_income, max_income.
    """
    df = get_predictions()
    if df.empty:
        return df
    if filters:
        if filters.get("department"):
            df = df[df["department"].isin(filters["department"])]
        if filters.get("risk_category"):
            df = df[df["risk_category"].isin(filters["risk_category"])]
        if filters.get("start_date"):
            df = df[
                pd.to_datetime(df["prediction_date"])
                >= pd.to_datetime(filters["start_date"])
            ]
        if filters.get("end_date"):
            df = df[
                pd.to_datetime(df["prediction_date"])
                <= pd.to_datetime(filters["end_date"])
            ]
        if filters.get("min_health") is not None:
            df = df[df["health_score"] >= filters["min_health"]]
        if filters.get("max_health") is not None:
            df = df[df["health_score"] <= filters["max_health"]]
        if filters.get("min_income") is not None:
            df = df[df["monthly_income"] >= filters["min_income"]]
        if filters.get("max_income") is not None:
            df = df[df["monthly_income"] <= filters["max_income"]]
    return df


# ---------------------------------------------------
# Simple intent detection (keyword based)
# ---------------------------------------------------
_INTENT_PATTERNS = {
    "top_risk": r"high[-\s]?risk|top[-\s]?risk|worst|worst risk|most risky",
    "department_risk": r"department.*risk|risk.*department",
    "high_risk_count": r"how many.*high.*risk|count.*high.*risk",
    "summary_today": r"today.*summary|summary.*today|today.*prediction",
    "low_health": r"low health|health score low|poor health",
    "dept_replacement": r"department.*replacement|replacement.*department",
    "retention_recommend": r"retention|recommendation|advice",
    "shap_explain": r"shap|explain.*shap|shap values",
    "overtime_attrition": r"overtime.*attrition|why.*overtime",
    "executive_summary": r"executive summary|summary for exec",
}


def detect_intent(user_msg: str) -> str:
    """Return a canonical intent key based on keyword matching.
    Falls back to ``general`` if no pattern matches.
    """
    msg = user_msg.lower()
    for intent, pattern in _INTENT_PATTERNS.items():
        if re.search(pattern, msg):
            return intent
    return "general"


# ---------------------------------------------------
# Local analytics fallback
# ---------------------------------------------------
def answer_locally(intent: str, df: pd.DataFrame):
    """Generate a textual answer (and optional Plotly figure) for a given intent.
    Returns ``(answer_str, fig_or_None)``.
    """
    if df.empty:
        return "No prediction data is available.", None
    try:
        if intent == "top_risk":
            top = df.nlargest(5, "prediction_probability")[
                ["employee_name", "department", "prediction_probability"]
            ]
            rows = "\n".join(
                [
                    f"- {row['employee_name'] or 'Unnamed'} ({row['department']}): {row['prediction_probability']*100:.2f}%"
                    for _, row in top.iterrows()
                ]
            )
            return f"**Top 5 highest‑risk employees**:\n{rows}", None
        if intent == "department_risk":
            dept = (
                df.groupby("department")["prediction_probability"].mean().reset_index()
            )
            fig = px.bar(
                dept,
                x="department",
                y="prediction_probability",
                title="Avg Attrition Probability by Department",
            )
            return "Average attrition probability by department:", fig
        if intent == "high_risk_count":
            cnt = (df["risk_category"].str.lower() == "high").sum()
            return f"There are **{cnt}** high‑risk employees in the database.", None
        if intent == "summary_today":
            today = datetime.now().date()
            today_df = df[pd.to_datetime(df["prediction_date"]).dt.date == today]
            total = len(today_df)
            high = (today_df["risk_category"].str.lower() == "high").sum()
            return (
                f"Today's predictions: **{total}** total, **{high}** high‑risk.",
                None,
            )
        if intent == "low_health":
            low = df[df["health_score"] <= 30][
                ["employee_name", "department", "health_score"]
            ]
            if low.empty:
                return "No employees with low health scores (≤30) found.", None
            rows = "\n".join(
                [
                    f"- {row['employee_name'] or 'Unnamed'} ({row['department']}): {row['health_score']}"
                    for _, row in low.iterrows()
                ]
            )
            return f"Employees with low health scores (≤30):\n{rows}", None
        if intent == "dept_replacement":
            cost = df.groupby("department")["replacement_cost"].sum().reset_index()
            fig = px.treemap(
                cost,
                path=["department"],
                values="replacement_cost",
                title="Replacement Cost by Department",
            )
            return "Total replacement cost per department:", fig
        if intent == "retention_recommend":
            # Simple rule‑based recommendation based on top risk factors
            high_risk = df[df["risk_category"].str.lower() == "high"]
            reasons = []
            if high_risk["overtime"].str.lower().eq("yes").any():
                reasons.append("• Reduce overtime workload.")
            if (high_risk["work_life_balance"] <= 2).any():
                reasons.append("• Introduce work‑life balance programs.")
            if (high_risk["job_satisfaction"] <= 2).any():
                reasons.append("• Conduct satisfaction surveys and address concerns.")
            if not reasons:
                reasons.append(
                    "• Review compensation and career progression opportunities."
                )
            return "**Retention recommendations**:\n" + "\n".join(reasons), None
        if intent == "shap_explain":
            # Provide a generic explanation
            txt = (
                "SHAP (SHapley Additive exPlanations) quantifies how each feature pushes the model "
                + "output higher or lower. Positive SHAP values increase the predicted attrition risk, "
                + "negative values decrease it. Summing all SHAP values (plus the base value) yields the "
                + "model’s probability. This helps identify the most influential factors for an individual employee."
            )
            return txt, None
        if intent == "overtime_attrition":
            pct = (df["overtime"].str.lower() == "yes").mean() * 100
            return (
                f"{pct:.1f}% of recorded predictions involve employees who work overtime. Historically, overtime correlates with higher attrition risk, as reflected in the model’s feature importance.",
                None,
            )
        if intent == "executive_summary":
            total = len(df)
            avg_prob = df["prediction_probability"].mean() * 100
            high = (df["risk_category"].str.lower() == "high").sum()
            return (
                f"**Executive Summary**\n- Total predictions: {total}\n- Average attrition probability: {avg_prob:.2f}%\n- High‑risk employees: {high}\n- Avg health score: {df['health_score'].mean():.1f}\n- Avg monthly income: ₹{df['monthly_income'].mean():,.0f}"
            ), None
        # General fallback
        return (
            "I’m sorry, I couldn’t identify a specific request. Try asking about risk, department, or summaries.",
            None,
        )
    except Exception as e:
        return f"An error occurred while processing the request: {e}", None


# ---------------------------------------------------
# Export helpers
# ---------------------------------------------------
def export_chat_as_markdown(messages):
    """Convert a list of ``{'role': 'user'/'assistant', 'content': str}`` to markdown string."""
    parts = []
    for m in messages:
        prefix = "**You:**" if m["role"] == "user" else "**Copilot:**"
        parts.append(f"{prefix}\n\n{m['content']}\n")
    return "\n---\n\n".join(parts)


def export_chat_as_txt(messages):
    return "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])


def export_chat_as_pdf(messages):
    """Generate a PDF BytesIO object containing the chat transcript."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 40
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "AI HR Copilot Conversation")
    y -= 30
    c.setFont("Helvetica", 12)
    for m in messages:
        lines = (m["content"] or "").split("\n")
        header = f"{m['role'].capitalize()}:"
        c.drawString(40, y, header)
        y -= 18
        for line in lines:
            if y < 40:
                c.showPage()
                y = height - 40
            c.drawString(50, y, line)
            y -= 14
        y -= 10
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


# ---------------------------------------------------
# Gemini wrapper with graceful fallback
# ---------------------------------------------------
def generate_with_gemini(prompt: str):
    """Call Gemini model; on quota/network errors return ``None``.
    The ``ai_hr_assistant.generate_hr_analysis`` already contains similar logic, but
    we keep a lightweight wrapper here to avoid circular imports.
    """
    try:
        import google.generativeai as genai
        import streamlit as st

        genai.configure(api_key=st.secrets.get("GEMINI_API_KEY", ""))
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        if response and hasattr(response, "text") and response.text:
            return response.text
    except Exception as e:
        # Any exception is treated as unavailable
        return None
    return None

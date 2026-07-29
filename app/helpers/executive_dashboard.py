from io import BytesIO

import pandas as pd
from database import get_predictions
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def load_data(filters=None):
    """Retrieve predictions from SQLite and apply optional filters.
    Filters dict can contain:
        department (list), risk_category (list), start_date (str), end_date (str),
        min_health (int), max_health (int), min_income (float), max_income (float)
    """
    df = get_predictions()
    if df.empty:
        return df

    # Robustly ensure all required numeric columns exist
    numeric_columns = [
        "prediction_probability",
        "monthly_income",
        "replacement_cost",
        "health_score",
        "job_satisfaction",
        "environment_satisfaction",
        "work_life_balance",
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if filters:
        if filters.get("department") and "department" in df.columns:
            df = df[df["department"].isin(filters["department"])]
        if filters.get("risk_category") and "risk_category" in df.columns:
            df = df[df["risk_category"].isin(filters["risk_category"])]
        if filters.get("start_date") and "prediction_date" in df.columns:
            df = df[
                pd.to_datetime(df["prediction_date"], errors="coerce")
                >= pd.to_datetime(filters["start_date"], errors="coerce")
            ]
        if filters.get("end_date") and "prediction_date" in df.columns:
            df = df[
                pd.to_datetime(df["prediction_date"], errors="coerce")
                <= pd.to_datetime(filters["end_date"], errors="coerce")
            ]
        if filters.get("min_health") is not None and "health_score" in df.columns:
            df = df[df["health_score"] >= filters["min_health"]]
        if filters.get("max_health") is not None and "health_score" in df.columns:
            df = df[df["health_score"] <= filters["max_health"]]
        if filters.get("min_income") is not None and "monthly_income" in df.columns:
            df = df[df["monthly_income"] >= filters["min_income"]]
        if filters.get("max_income") is not None and "monthly_income" in df.columns:
            df = df[df["monthly_income"] <= filters["max_income"]]
    return df


def compute_kpis(df: pd.DataFrame):
    total = len(df)
    high = (
        (df["risk_category"].str.lower() == "high").sum()
        if "risk_category" in df.columns
        else 0
    )
    medium = (
        (df["risk_category"].str.lower() == "medium").sum()
        if "risk_category" in df.columns
        else 0
    )
    low = (
        (df["risk_category"].str.lower() == "low").sum()
        if "risk_category" in df.columns
        else 0
    )
    avg_prob = (
        df["prediction_probability"].mean()
        if total and "prediction_probability" in df.columns
        else 0
    )
    avg_health = (
        df["health_score"].mean() if total and "health_score" in df.columns else 0
    )
    avg_income = (
        df["monthly_income"].mean() if total and "monthly_income" in df.columns else 0
    )
    total_cost = (
        df["replacement_cost"].sum()
        if total and "replacement_cost" in df.columns
        else 0
    )
    return {
        "total": total,
        "high": high,
        "medium": medium,
        "low": low,
        "avg_prob": avg_prob,
        "avg_health": avg_health,
        "avg_income": avg_income,
        "total_cost": total_cost,
    }


def compute_insights(df: pd.DataFrame):
    insights = []
    if df.empty:
        return insights

    if "department" in df.columns and "prediction_probability" in df.columns:
        dept_risk = df.groupby("department")["prediction_probability"].mean()
        if not dept_risk.empty:
            highest_risk_dept = dept_risk.idxmax()
            insights.append(
                f"**Highest risk department:** {highest_risk_dept} (avg prob {dept_risk.max():.2%})"
            )

    if "department" in df.columns and "health_score" in df.columns:
        dept_health = df.groupby("department")["health_score"].mean()
        if not dept_health.empty:
            best_health_dept = dept_health.idxmax()
            insights.append(
                f"**Department with best health score:** {best_health_dept} (avg health {dept_health.max():.1f})"
            )

    if "prediction_probability" in df.columns:
        insights.append(
            f"**Average attrition probability:** {df['prediction_probability'].mean():.2%}"
        )

    if "replacement_cost" in df.columns:
        insights.append(
            f"**Average replacement cost:** ₹{df['replacement_cost'].mean():,.0f}"
        )

    if "risk_category" in df.columns and not df["risk_category"].empty:
        common_risk = df["risk_category"].mode()[0]
        insights.append(f"**Most common risk category:** {common_risk}")

    if (
        "monthly_income" in df.columns
        and "department" in df.columns
        and not df["monthly_income"].isnull().all()
    ):
        highest_income_row = df.loc[df["monthly_income"].idxmax()]
        insights.append(
            f"**Highest income employee predicted:** ₹{highest_income_row['monthly_income']:,.0f} in {highest_income_row['department']}"
        )

    if (
        "health_score" in df.columns
        and "department" in df.columns
        and not df["health_score"].isnull().all()
    ):
        lowest_health_row = df.loc[df["health_score"].idxmin()]
        insights.append(
            f"**Lowest health score employee:** {lowest_health_row['health_score']} in {lowest_health_row['department']}"
        )

    return insights


def generate_pdf_summary(kpis: dict, insights: list):
    """Create a PDF binary buffer with KPI summary and insights.
    Returns BytesIO object ready for download.
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 40
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "Executive Dashboard Summary")
    y -= 30
    c.setFont("Helvetica", 12)
    for key, value in kpis.items():
        line = f"{key.replace('_', ' ').title()}: {value if not isinstance(value, float) else f'{value:,.2f}'}"
        c.drawString(40, y, line)
        y -= 20
    y -= 10
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "Key Insights")
    y -= 25
    c.setFont("Helvetica", 12)
    for insight in insights:
        c.drawString(40, y, insight)
        y -= 18
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

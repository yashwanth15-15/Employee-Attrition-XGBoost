import os
import sys

import requests
import streamlit as st

# Add the app directory to sys.path so modules like api_client can be imported from any page
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import create_table
from helpers.metrics_calculator import get_model_metrics
from logger import logger
from api_client import BACKEND_URL

create_table()

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="assets/logo.png",
    initial_sidebar_state="expanded",
    layout="centered",
)
logger.info("Streamlit Application Started")

acc, f1 = get_model_metrics()

# Try to display logo if available
if os.path.exists("assets/logo.png"):
    st.image("assets/logo.png", width=80)

st.title("Employee Attrition Prediction Platform")
st.markdown("### Predict Employee Turnover Using Machine Learning & HR Analytics")

st.markdown("---")

# Backend Status Check
st.markdown("#### System Status")
with st.container():
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/health", timeout=3)
        if response.status_code == 200:
            st.success("✅ Backend API is running and connected.")
            logger.info("Backend connection successful.")
        else:
            st.warning(
                f"⚠️ Backend API returned unexpected status: {response.status_code}"
            )
            logger.warning(f"Backend API status: {response.status_code}")
    except Exception as e:
        st.error("❌ Failed to connect to Backend API. Is it running?")
        logger.error(f"Backend connection failed: {str(e)}")

st.markdown("---")

with st.expander("ℹ️ About this Project", expanded=True):
    st.markdown("### Project Overview")
    st.info(
        "**Employee Attrition Prediction & HR Analytics Platform**\n\n"
        "An end-to-end AI-powered decision support system that predicts employee attrition risk, "
        "explains predictions using SHAP Explainable AI, and provides actionable HR insights through "
        "interactive dashboards and REST APIs."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Key Features")
        st.markdown(
            "• Employee Attrition Prediction\n"
            "• Batch Prediction\n"
            "• SHAP Explainable AI\n"
            "• HR Recommendation Engine\n"
            "• Executive Dashboard\n"
            "• Department Analytics\n"
            "• Employee Comparison\n"
            "• Workforce Planning\n"
            "• Prediction History\n"
            "• PDF Report Generation\n"
            "• REST API Integration\n"
            "• Docker Ready"
        )
    with col2:
        st.markdown("### Technology Stack")
        st.markdown(
            "**Frontend:** Streamlit\n\n"
            "**Backend:** FastAPI\n\n"
            "**Machine Learning:** XGBoost\n\n"
            "**Explainable AI:** SHAP\n\n"
            "**Database:** SQLite\n\n"
            "**Visualization:** Plotly\n\n"
            "**Deployment:** Docker, Render\n\n"
            "**Language:** Python"
        )

    st.markdown("---")

    st.markdown("### Architecture")
    st.markdown(
        "<div style='text-align: center; font-family: monospace;'>"
        "User <br>↓<br>"
        "Streamlit Dashboard <br>↓<br>"
        "FastAPI REST API <br>↓<br>"
        "XGBoost Prediction Engine <br>↓<br>"
        "SHAP Explanation Engine <br>↓<br>"
        "SQLite Database"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### Model Summary")
        st.markdown(
            "**Algorithm:** XGBoost Classifier\n\n"
            "**Prediction:** Employee Attrition Risk\n\n"
            "**Output:**\n"
            "• Probability\n"
            "• Risk Level\n"
            "• SHAP Explanation\n"
            "• HR Recommendations"
        )
    with col4:
        st.markdown("### Project Statistics")
        st.metric("Number of Features", "34")
        st.metric("Prediction Accuracy", f"{acc*100:.2f}%")
        st.metric("ML Algorithm", "XGBoost")
        st.metric("REST Endpoints", "5+")
        st.metric("Dashboard Pages", "13")
        st.metric("Model Explainability", "SHAP")


st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey; font-size: 0.9em;'>"
    "<b>Version:</b> 1.0.0 | <b>Model:</b> XGBoost Classifier<br>"
    "Built with Streamlit + FastAPI | Developer: B. Yashwanth"
    "</div>",
    unsafe_allow_html=True,
)
logger.info("Main dashboard rendered successfully.")

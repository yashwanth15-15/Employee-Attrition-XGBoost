import os
import requests
import streamlit as st
from fastapi import status

BACKEND_URL = os.getenv(
    "BACKEND_URL", "http://127.0.0.1:8000"
)


def _get_headers():
    headers = {}
    if "access_token" in st.session_state:
        headers["Authorization"] = f"Bearer {st.session_state['access_token']}"
    return headers


def _handle_response(response):
    if response.status_code == status.HTTP_401_UNAUTHORIZED:
        st.session_state.pop("access_token", None)
        st.session_state.pop("username", None)
        st.session_state.pop("role", None)
        st.error("Session expired or unauthorized. Please log in again.")
        st.stop()
    elif response.status_code == status.HTTP_403_FORBIDDEN:
        st.error("You do not have permission to access this page.")
        st.stop()
    
    response.raise_for_status()
    return response.json()


def check_backend_health():
    """Check if the FastAPI backend is reachable."""
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/health", timeout=3)
        return response.status_code == 200
    except requests.RequestException:
        return False


def predict_employee(features_dict):
    """Call the predict endpoint."""
    url = f"{BACKEND_URL}/api/v1/predict"
    response = requests.post(url, json=features_dict, headers=_get_headers(), timeout=10)
    return _handle_response(response)


def simulate_prediction(base_features, modified_features):
    """Call the simulate endpoint."""
    url = f"{BACKEND_URL}/api/v1/simulate"
    payload = {"base_features": base_features, "modified_features": modified_features}
    response = requests.post(url, json=payload, headers=_get_headers(), timeout=10)
    return _handle_response(response)


def get_analytics_summary():
    """Call the analytics summary endpoint."""
    url = f"{BACKEND_URL}/api/v1/analytics/summary"
    response = requests.get(url, headers=_get_headers(), timeout=10)
    return _handle_response(response)


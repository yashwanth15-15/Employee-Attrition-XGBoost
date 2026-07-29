"""
Constants used across the FastAPI backend.
Contains default values for features, limits, and other hardcoded logic.
"""

from typing import Any, Dict

# Feature Default Values (ML Service)
DEFAULT_PERFORMANCE_RATING = 3
DEFAULT_JOB_INVOLVEMENT = 3
DEFAULT_EDUCATION = 3
DEFAULT_NUM_COMPANIES_WORKED = 1
DEFAULT_RELATIONSHIP_SATISFACTION = 3
DEFAULT_JOB_LEVEL = 1
DEFAULT_SALARY_HIKE = 11

# Recommendation Engine Defaults
DEFAULT_RECOMMENDATION_FEATURES: Dict[str, Any] = {
    "Age": 30,
    "Gender": "Male",
    "Department": "Sales",
    "Monthly Income": 5000,
    "Marital Status": "Single",
    "OverTime": "No",
    "Years At Company": 0,
    "Total Working Years": 0,
    "Job Satisfaction": 3,
    "Environment Satisfaction": 3,
    "Work Life Balance": 3,
    "Years Since Last Promotion": 0,
    "Training Times Last Year": 2,
    "Business Travel": "Non-Travel",
    "Distance From Home": 5,
    "Performance Rating": 3,
    "Stock Option Level": 0,
    "Years In Current Role": 0,
}

# SHAP Configuration
MAX_TOP_FEATURES = 5

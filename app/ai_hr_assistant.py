import streamlit as st
from helpers.risk_calculator import calculate_risk

GEMINI_AVAILABLE = False
model = None

try:
    import google.generativeai as genai
    
    # Safely get API key
    api_key = st.secrets.get("GEMINI_API_KEY")
    
    if api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False


def generate_hr_analysis(employee, probability):
    """
    Generate professional HR recommendations based on
    employee information and predicted attrition probability.
    """
    if not GEMINI_AVAILABLE:
        return None

    # Determine Risk Level
    risk_level = calculate_risk(probability)

    prompt = f"""
You are a Senior HR Analytics Consultant.

Employee Details:
{employee}

Predicted Attrition Probability:
{probability:.2%}

Overall Risk Level:
{risk_level}

Generate a professional HR report.

IMPORTANT RULES

If Risk Level is LOW:
- Mention employee strengths first.
- Recommend preventive actions only.
- Do NOT make the employee sound likely to resign.

If Risk Level is MEDIUM:
- Explain moderate concerns.
- Suggest retention strategies.

If Risk Level is HIGH:
- Recommend urgent HR interventions.

Return ONLY these sections:

# 🔍 Overall Risk

# 📌 Key Risk Factors

# 🎯 HR Recommendations

# 📈 Retention Strategy

# 💼 Business Impact

# ⭐ Positive Employee Strengths

Rules:
- Keep the report under 250 words.
- Use professional HR language.
- Use Markdown headings.
- Do not repeat employee details.
"""

    try:

        response = model.generate_content(prompt)

        if (
            response
            and hasattr(response, "text")
            and response.text
        ):
            return response.text

        return None

    except Exception:
        return None
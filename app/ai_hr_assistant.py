import streamlit as st
import google.generativeai as genai

# Configure Gemini API
genai.configure(
    api_key=st.secrets["GEMINI_API_KEY"]
)

# Load Gemini Model
model = genai.GenerativeModel("gemini-2.5-flash")


def generate_hr_analysis(employee, probability):
    """
    Generate professional HR recommendations based on
    employee information and predicted attrition probability.
    """

    # Determine Risk Level
    if probability < 0.20:
        risk_level = "Low"
    elif probability < 0.50:
        risk_level = "Medium"
    else:
        risk_level = "High"

    prompt = f"""
You are a Senior HR Analytics Consultant.

Employee Details:
{employee}

Predicted Attrition Probability:
{probability:.2%}

Overall Risk Level:
{risk_level}

Generate a professional HR report using Markdown.

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

Return ONLY the following sections.

# 🔍 Overall Risk

Give a short summary (2-3 lines).

# 📌 Key Risk Factors

List only 3-5 bullet points.

# 🎯 HR Recommendations

List only 4-6 bullet points.

# 📈 Retention Strategy

List only 3-5 bullet points.

# 💼 Business Impact

Explain in 2-3 sentences.

# ⭐ Positive Employee Strengths

Mention positive aspects of the employee profile.

Rules:
- Keep the report below 250 words.
- Use professional HR language.
- Use Markdown headings.
- Do not repeat employee details.
- Keep recommendations realistic.
"""

    try:

        response = model.generate_content(prompt)

        if response.text:
            return response.text

        return "No response generated."

    except Exception as e:

        return f"""
## ❌ AI Error

Unable to generate HR analysis.

Error Details:

{str(e)}
"""
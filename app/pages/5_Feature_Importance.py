import pickle
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Feature Importance", layout="wide")

st.title("🌟 Feature Importance Dashboard")
st.markdown("""
This dashboard highlights which factors the XGBoost model relies on most when predicting employee attrition. 
Understanding these features allows HR to focus on the root causes of turnover rather than symptoms.
""")

# ==========================================
# DICTIONARY / METADATA
# ==========================================
FEATURE_METADATA = {
    "MonthlyIncome": {
        "category": "Compensation",
        "meaning": "Lower salary often increases resignation probability.",
        "recommendation": "Review salary competitiveness against market benchmarks.",
    },
    "OverTime": {
        "category": "Work Environment",
        "meaning": "Consistent overtime leads to burnout and dissatisfaction.",
        "recommendation": "Monitor working hours and consider hiring additional resources or redistributing workload.",
    },
    "Age": {
        "category": "Personal",
        "meaning": "Younger employees tend to change jobs more frequently for career progression.",
        "recommendation": "Provide clear career advancement paths and mentorship for younger demographics.",
    },
    "DailyRate": {
        "category": "Compensation",
        "meaning": "Compensation structures influence retention.",
        "recommendation": "Evaluate the fairness of daily rates across similar roles.",
    },
    "TotalWorkingYears": {
        "category": "Career Growth",
        "meaning": "Total experience indicates career maturity; mid-career employees often seek major shifts.",
        "recommendation": "Offer leadership training and upskilling for mid-career employees.",
    },
    "YearsAtCompany": {
        "category": "Career Growth",
        "meaning": "Tenure at the company affects loyalty; early years have higher attrition risk.",
        "recommendation": "Enhance onboarding and first-year engagement programs.",
    },
    "DistanceFromHome": {
        "category": "Work Environment",
        "meaning": "Long commutes cause stress and degrade work-life balance.",
        "recommendation": "Offer remote work options, flexible hours, or commuter benefits.",
    },
    "HourlyRate": {
        "category": "Compensation",
        "meaning": "Variations in hourly pay affect perceived fairness.",
        "recommendation": "Ensure pay equity among hourly workers.",
    },
    "MonthlyRate": {
        "category": "Compensation",
        "meaning": "Variations in monthly rates affect overall compensation perception.",
        "recommendation": "Regularly review total compensation packages.",
    },
    "NumCompaniesWorked": {
        "category": "Personal",
        "meaning": "Employees with a history of frequent job changes may continue the pattern.",
        "recommendation": "Screen for cultural fit and long-term goals during hiring.",
    },
    "JobRole": {
        "category": "Job Characteristics",
        "meaning": "Specific roles (e.g., Sales Reps) naturally experience higher turnover.",
        "recommendation": "Tailor retention strategies to high-risk roles.",
    },
    "JobSatisfaction": {
        "category": "Work Environment",
        "meaning": "Direct measure of employee happiness with their daily tasks.",
        "recommendation": "Conduct regular pulse surveys and act on feedback immediately.",
    },
    "EnvironmentSatisfaction": {
        "category": "Work Environment",
        "meaning": "Dissatisfaction with the physical or cultural environment drives talent away.",
        "recommendation": "Improve workspace conditions and foster a positive team culture.",
    },
    "YearsInCurrentRole": {
        "category": "Career Growth",
        "meaning": "Stagnation in one role leads to boredom and lack of progression.",
        "recommendation": "Encourage internal mobility and job rotation.",
    },
    "YearsWithCurrManager": {
        "category": "Work Environment",
        "meaning": "The relationship with a direct manager is a primary reason people leave or stay.",
        "recommendation": "Provide leadership training for managers with high team turnover.",
    },
    "WorkLifeBalance": {
        "category": "Work Environment",
        "meaning": "Poor work-life balance is unsustainable for long-term employment.",
        "recommendation": "Mandate vacation time and discourage after-hours communication.",
    },
    "TrainingTimesLastYear": {
        "category": "Career Growth",
        "meaning": "Lack of training indicates a lack of investment in employee development.",
        "recommendation": "Increase professional development budgets and training opportunities.",
    },
    "StockOptionLevel": {
        "category": "Compensation",
        "meaning": "Lack of equity or stock options reduces long-term financial ties to the company.",
        "recommendation": "Expand stock option eligibility to increase ownership mentality.",
    },
    "YearsSinceLastPromotion": {
        "category": "Career Growth",
        "meaning": "Delayed promotions signal a lack of career advancement.",
        "recommendation": "Review promotion cycles and ensure deserving employees are recognized.",
    },
    "JobInvolvement": {
        "category": "Work Environment",
        "meaning": "Low involvement indicates disengagement and apathy.",
        "recommendation": "Assign meaningful projects and increase employee autonomy.",
    },
    "MaritalStatus": {
        "category": "Personal",
        "meaning": "Marital status correlates with stability and financial needs.",
        "recommendation": "Offer family-friendly benefits and flexible schedules.",
    },
    "BusinessTravel": {
        "category": "Job Characteristics",
        "meaning": "Frequent travel can cause burnout and disrupt personal lives.",
        "recommendation": "Optimize travel requirements and offer recovery days post-travel.",
    },
    "JobLevel": {
        "category": "Job Characteristics",
        "meaning": "Entry-level positions usually have higher turnover than senior roles.",
        "recommendation": "Create fast-track programs for high-potential junior staff.",
    },
    "EducationField": {
        "category": "Personal",
        "meaning": "Alignment between education and role impacts job satisfaction.",
        "recommendation": "Ensure roles utilize employees' educational strengths.",
    },
    "Department": {
        "category": "Job Characteristics",
        "meaning": "Different departments have different sub-cultures and pressures.",
        "recommendation": "Investigate sub-cultures in departments with unusually high attrition.",
    },
    "Education": {
        "category": "Personal",
        "meaning": "Higher education levels may demand faster career progression.",
        "recommendation": "Provide continuous learning and advanced challenges.",
    },
    "Gender": {
        "category": "Personal",
        "meaning": "Gender dynamics can highlight inclusion or pay equity issues.",
        "recommendation": "Audit pay equity and ensure a highly inclusive culture.",
    },
    "PerformanceRating": {
        "category": "Career Growth",
        "meaning": "High performers expect rewards; low performers may be managed out.",
        "recommendation": "Ensure top performers are adequately recognized and compensated.",
    },
    "RelationshipSatisfaction": {
        "category": "Work Environment",
        "meaning": "Poor relationships with colleagues reduce workplace attachment.",
        "recommendation": "Organize team-building activities and foster collaboration.",
    },
}


def get_meta(feature):
    base = feature.split("_")[0] if "_" in feature else feature
    return FEATURE_METADATA.get(
        base,
        {
            "category": "Other",
            "meaning": f"The value of '{feature}' impacts the model's prediction.",
            "recommendation": f"Monitor the distribution of '{feature}' among employees.",
        },
    )


# ==========================================
# DATA & MODEL LOADING
# ==========================================
@st.cache_resource
def load_feature_importance():
    try:
        with open("models/final_xgboost_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("models/final_features.pkl", "rb") as f:
            features = pickle.load(f)

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            df = pd.DataFrame({"Feature": features, "Importance": importances})
            df = df.sort_values(by="Importance", ascending=False).reset_index(drop=True)
            df["Percentage"] = (df["Importance"] / df["Importance"].sum()) * 100
            df["Cumulative (%)"] = df["Percentage"].cumsum()
            df["Rank"] = df.index + 1
            df["Category"] = df["Feature"].apply(lambda x: get_meta(x)["category"])
            df["Business Meaning"] = df["Feature"].apply(
                lambda x: get_meta(x)["meaning"]
            )
            df["HR Recommendation"] = df["Feature"].apply(
                lambda x: get_meta(x)["recommendation"]
            )
            return df, model
        return None, None
    except FileNotFoundError:
        return None, None


with st.spinner("Loading Feature Importance Data..."):
    df_fi, model = load_feature_importance()

if df_fi is None or model is None:
    st.error(
        "❌ Model artifacts or feature importances missing! Ensure models exist in the 'models/' directory."
    )
    st.stop()

# ==========================================
# KPI CARDS
# ==========================================
st.markdown("### 📌 Summary")

top_feature = df_fi.iloc[0]["Feature"]
top_5_contrib = df_fi.head(5)["Percentage"].sum()
avg_importance = df_fi["Importance"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Features", f"{len(df_fi)}")
c2.metric("Most Important Feature", top_feature)
c3.metric("Avg Feature Importance", f"{avg_importance:.4f}")
c4.metric("Top 5 Contribution", f"{top_5_contrib:.1f}%")
c5.metric("Model Type", "XGBoost")

st.divider()

# ==========================================
# VISUALIZATIONS
# ==========================================
col_v1, col_v2 = st.columns(2)

# 1. Top 20 Feature Importance
with col_v1:
    st.subheader("1. Top 20 Feature Importance")
    top20 = df_fi.head(20).sort_values(by="Importance", ascending=True)
    fig_bar = px.bar(
        top20,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 20 Drivers of Attrition",
    )
    st.plotly_chart(fig_bar, width="stretch")

# 2. Feature Importance Treemap
with col_v2:
    st.subheader("2. Importance by Category (Treemap)")
    fig_tree = px.treemap(
        df_fi,
        path=["Category", "Feature"],
        values="Percentage",
        title="Hierarchical Feature Contribution",
    )
    st.plotly_chart(fig_tree, width="stretch")

st.divider()

col_v3, col_v4 = st.columns(2)

# 3. Cumulative Importance
with col_v3:
    st.subheader("3. Cumulative Importance")
    fig_line = px.line(
        df_fi,
        x="Rank",
        y="Cumulative (%)",
        title="Cumulative Feature Contribution",
        markers=True,
    )
    fig_line.add_hline(
        y=80, line_dash="dash", line_color="red", annotation_text="80% Threshold"
    )
    st.plotly_chart(fig_line, width="stretch")

# 5. Feature Importance Distribution
with col_v4:
    st.subheader("5. Importance Distribution")
    fig_hist = px.histogram(
        df_fi,
        x="Importance",
        nbins=20,
        title="Distribution of Feature Importance Scores",
    )
    st.plotly_chart(fig_hist, width="stretch")

st.divider()

# 4. Top 10 Features Table
st.subheader("4. Top 10 Features Breakdown")
top10_df = df_fi.head(10)[
    ["Rank", "Feature", "Importance", "Percentage", "Business Meaning"]
]
top10_df["Percentage"] = top10_df["Percentage"].apply(lambda x: f"{x:.1f}%")
top10_df["Importance"] = top10_df["Importance"].apply(lambda x: f"{x:.4f}")
st.dataframe(top10_df, hide_index=True, width="stretch")

st.divider()

col_exp, col_cat = st.columns([1, 1])

# 6. Interactive Feature Explorer
with col_exp:
    st.subheader("6. Interactive Feature Explorer")
    selected_feature = st.selectbox(
        "Select a Feature to inspect:", df_fi["Feature"].tolist()
    )
    feat_data = df_fi[df_fi["Feature"] == selected_feature].iloc[0]

    st.info(f"**Rank:** #{feat_data['Rank']} out of {len(df_fi)}")
    st.success(
        f"**Importance Score:** {feat_data['Importance']:.4f} ({feat_data['Percentage']:.1f}% contribution)"
    )
    st.warning(f"**Category:** {feat_data['Category']}")
    st.markdown(f"**🏢 Business Description:** {feat_data['Business Meaning']}")
    st.markdown(f"**💡 HR Impact:** {feat_data['HR Recommendation']}")

# Feature Categories Pie Chart
with col_cat:
    st.subheader("Importance by Business Category")
    cat_df = df_fi.groupby("Category")["Percentage"].sum().reset_index()
    fig_pie = px.pie(
        cat_df,
        names="Category",
        values="Percentage",
        hole=0.4,
        title="Category-wise Contribution",
    )
    st.plotly_chart(fig_pie, width="stretch")

st.divider()

# ==========================================
# BUSINESS INTERPRETATION (TOP 10)
# ==========================================
st.subheader("🏢 Deep Dive: Top Drivers of Attrition")
st.markdown(
    "Below are the automated business interpretations for the top 10 most influential features driving employee turnover."
)

for i in range(10):
    row = df_fi.iloc[i]
    with st.expander(f"#{row['Rank']}: {row['Feature']} ({row['Percentage']:.1f}%)"):
        st.write(f"**Why it matters:** {row['Business Meaning']}")
        st.write(f"**HR Recommendation:** {row['HR Recommendation']}")

st.divider()

# ==========================================
# ADVANCED ANALYSIS
# ==========================================
st.subheader("🔬 Advanced Analysis Summary")

top_3_feat = df_fi.head(3)["Feature"].tolist()
low_imp = df_fi[df_fi["Percentage"] < 1.0]

c_adv1, c_adv2 = st.columns(2)
with c_adv1:
    st.info(
        f"**Most Influential Features:** The top 3 drivers dominating the model's decisions are **{', '.join(top_3_feat)}**. HR should prioritize initiatives targeting these areas immediately."
    )
with c_adv2:
    st.info(
        f"**Low Importance Features:** There are **{len(low_imp)}** features contributing less than 1% each to the model's predictions. These have minimal impact on overall attrition risk."
    )

st.divider()

# ==========================================
# EXPORT
# ==========================================
st.subheader("📥 Export Feature Importance")
col_csv1, col_csv2 = st.columns(2)

with col_csv1:
    csv_full = (
        df_fi[["Rank", "Feature", "Importance", "Percentage", "Category"]]
        .to_csv(index=False)
        .encode("utf-8")
    )
    st.download_button(
        label="Download Feature Ranking (CSV)",
        data=csv_full,
        file_name=f"employee_attrition_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

with col_csv2:
    csv_business = (
        df_fi[["Feature", "Business Meaning", "HR Recommendation"]]
        .to_csv(index=False)
        .encode("utf-8")
    )
    st.download_button(
        label="Download Business Interpretations (CSV)",
        data=csv_business,
        file_name=f"employee_attrition_feature_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

st.caption(
    "Generated by Employee Attrition Prediction System | Dynamic Feature Importance Module"
)

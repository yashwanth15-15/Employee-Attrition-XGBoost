import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="SHAP Explainability", layout="wide")

st.title("🧩 Explainable AI (SHAP)")
st.markdown("""
**What is SHAP?**  
SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain the output of any machine learning model. 
It breaks down a prediction to show the impact of each feature. This dashboard allows you to peek inside the "black box" 
of the XGBoost model to understand exactly *why* it predicts attrition globally and for individual employees.
""")


# ==========================================
# DICTIONARY / METADATA
# ==========================================
def get_business_interpretation(feature, shap_val, feat_val):
    direction = "increases" if shap_val > 0 else "lowers"
    impact = "attrition risk"

    # Generic templates based on SHAP direction
    if shap_val > 0:
        return f"A value of '{feat_val}' for **{feature}** {direction} the probability of attrition."
    else:
        return f"A value of '{feat_val}' for **{feature}** {direction} the probability of attrition, acting as a retention stabilizer."


# ==========================================
# DATA, MODEL, & SHAP LOADING
# ==========================================
@st.cache_resource
def load_model_and_explainer():
    try:
        with open("models/final_xgboost_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("models/final_encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        with open("models/final_features.pkl", "rb") as f:
            features = pickle.load(f)

        explainer = shap.TreeExplainer(model)
        return model, encoders, features, explainer
    except FileNotFoundError:
        return None, None, None, None


@st.cache_data
def load_and_compute_shap(_encoders, features, _explainer):
    try:
        df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    except FileNotFoundError:
        return None, None, None, None

    df_encoded = df.copy()
    for col, enc in _encoders.items():
        if col in df_encoded.columns and col != "Attrition":
            df_encoded[col] = enc.transform(df_encoded[col].astype(str))

    df_encoded["Attrition"] = df_encoded["Attrition"].apply(
        lambda x: 1 if x == "Yes" else 0
    )

    X = df_encoded[features]
    y = df_encoded["Attrition"]

    # Evaluate on Test Set
    _, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Original unencoded test data for display
    X_test_display = df.loc[X_test.index, features]

    # Compute SHAP values
    shap_values = _explainer.shap_values(X_test)
    base_value = _explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = base_value[0]

    return X_test, X_test_display, shap_values, base_value


model, encoders, features, explainer = load_model_and_explainer()

if model is None or explainer is None:
    st.error(
        "❌ Model artifacts missing! Ensure models exist in the 'models/' directory."
    )
    st.stop()

with st.spinner("Loading SHAP explanations..."):
    X_test, X_test_display, shap_values, base_value = load_and_compute_shap(
        encoders, features, explainer
    )

if X_test is None:
    st.error(
        "❌ Dataset missing! Ensure 'data/WA_Fn-UseC_-HR-Employee-Attrition.csv' exists."
    )
    st.stop()

# Compute mean absolute SHAP values for global importance
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({"Feature": features, "Mean |SHAP|": mean_abs_shap})
shap_df = shap_df.sort_values(by="Mean |SHAP|", ascending=False).reset_index(drop=True)

# ==========================================
# KPI CARDS
# ==========================================
st.markdown("### 📌 Explainability Overview")
top_shap_feat = shap_df.iloc[0]["Feature"]
avg_shap = np.mean(mean_abs_shap)

kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)
kpi_c1.metric("Model", "XGBoost")
kpi_c2.metric("Total Features", f"{len(features)}")
kpi_c3.metric("Explained Samples", f"{len(X_test)}")
kpi_c4.metric("Average |SHAP|", f"{avg_shap:.4f}")
kpi_c5.metric("Most Influential Feature", top_shap_feat)

st.divider()

# ==========================================
# GLOBAL EXPLAINABILITY VISUALIZATIONS
# ==========================================
st.header("🌍 Global Explanations")
st.markdown(
    "Global explanations show how the model behaves overall across all employees."
)

col_g1, col_g2 = st.columns(2)

# 1. SHAP Summary Plot (Beeswarm via Pyplot)
with col_g1:
    st.subheader("1. SHAP Summary Plot (Beeswarm)")
    # SHAP creates its own figure
    shap.summary_plot(
        shap_values, X_test, plot_type="dot", show=False, feature_names=features
    )
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)
    st.info(
        "**How to read:** Each dot is an employee. Red indicates a high feature value, blue indicates a low value. Dots to the right push the model toward Attrition (Yes)."
    )

# 2. SHAP Feature Importance (Bar)
with col_g2:
    st.subheader("2. SHAP Feature Importance")
    top_shap = shap_df.head(15).sort_values(by="Mean |SHAP|", ascending=True)
    fig_bar = px.bar(
        top_shap,
        x="Mean |SHAP|",
        y="Feature",
        orientation="h",
        title="Top 15 Features by Mean |SHAP|",
    )
    st.plotly_chart(fig_bar, width="stretch")
    st.info(
        "**Why it matters:** Unlike standard XGBoost importance, SHAP importance measures the average absolute impact each feature has on the final predicted probability."
    )

# 3. Dependence Plot
st.subheader("3. Feature Dependence Plot")
col_d1, col_d2 = st.columns([1, 3])
with col_d1:
    dep_feat = st.selectbox("Select a Feature to analyze:", features)
with col_d2:
    feat_idx = features.index(dep_feat)
    dep_df = pd.DataFrame(
        {"Feature Value": X_test[dep_feat], "SHAP Value": shap_values[:, feat_idx]}
    )
    try:
        fig_dep = px.scatter(
            dep_df,
            x="Feature Value",
            y="SHAP Value",
            trendline="lowess",
            title=f"Dependence Plot for {dep_feat}",
        )
    except (ImportError, ModuleNotFoundError):
        fig_dep = px.scatter(
            dep_df,
            x="Feature Value",
            y="SHAP Value",
            title=f"Dependence Plot for {dep_feat}",
        )
    fig_dep.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_dep, width="stretch")
    st.info(
        f"**Interpretation:** Shows how varying **{dep_feat}** directly impacts the prediction. Points above 0 increase attrition risk; points below 0 decrease it."
    )

st.divider()

# ==========================================
# LOCAL EXPLAINABILITY (EMPLOYEE EXPLORER)
# ==========================================
st.header("👤 Local Explanations (Interactive Employee Explorer)")
st.markdown(
    "Select a specific employee to see exactly why the model predicted they will stay or leave."
)

emp_indices = X_test.index.tolist()
selected_emp_idx = st.selectbox("Select Employee ID (Test Set Index):", emp_indices)

# Extract Local Data
loc_idx = emp_indices.index(selected_emp_idx)
loc_x_encoded = X_test.iloc[loc_idx]
loc_x_display = X_test_display.iloc[loc_idx]
loc_shap = shap_values[loc_idx]

# Predict
prob = model.predict_proba(loc_x_encoded.values.reshape(1, -1))[0, 1]
pred_class = "Attrition (Yes)" if prob >= 0.5 else "Retention (No)"

# Local SHAP DataFrame
loc_df = pd.DataFrame(
    {
        "Feature": features,
        "Value": loc_x_display.values.astype(str),
        "SHAP Value": loc_shap,
    }
)
loc_df["Contribution Direction"] = loc_df["SHAP Value"].apply(
    lambda x: "Increases Risk 📈" if x > 0 else "Lowers Risk 📉"
)
loc_df["Absolute SHAP"] = np.abs(loc_df["SHAP Value"])
loc_df = loc_df.sort_values(by="Absolute SHAP", ascending=False).drop(
    columns=["Absolute SHAP"]
)

# Employee KPIs
st.markdown(
    f"#### 🎯 Prediction for Employee **{selected_emp_idx}**: **{pred_class}** (Probability: {prob*100:.1f}%)"
)

col_l1, col_l2 = st.columns([2, 1])

# 4. Waterfall Plot (Plotly Equivalent)
with col_l1:
    st.subheader("4. Prediction Breakdown (Waterfall)")

    # Prepare waterfall data (Top 10 features + Rest)
    wf_df = loc_df.head(10).copy()
    rest_shap = loc_df.iloc[10:]["SHAP Value"].sum()
    if len(loc_df) > 10:
        rest_row = pd.DataFrame(
            [{"Feature": "All Other Features", "Value": "-", "SHAP Value": rest_shap}]
        )
        wf_df = pd.concat([wf_df, rest_row], ignore_index=True)

    # Reverse to build waterfall from bottom up
    wf_df = wf_df.iloc[::-1]

    measure = ["relative"] * len(wf_df)

    fig_wf = go.Figure(
        go.Waterfall(
            name="Employee Breakdown",
            orientation="h",
            measure=measure,
            y=wf_df["Feature"],
            x=wf_df["SHAP Value"],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "blue"}},
            increasing={"marker": {"color": "red"}},
        )
    )
    fig_wf.update_layout(
        title="Feature Contributions to Prediction Log-Odds", waterfallgap=0.3
    )
    st.plotly_chart(fig_wf, width="stretch")

# 6. Local Explanation Table
with col_l2:
    st.subheader("Local Explanation Table")
    st.dataframe(
        loc_df[["Feature", "Value", "SHAP Value", "Contribution Direction"]].head(12),
        hide_index=True,
        width="stretch",
    )

# Employee Business Interpretation
st.subheader("💼 Personalized HR Insight")
top_pos = loc_df[loc_df["SHAP Value"] > 0].head(2)
top_neg = loc_df[loc_df["SHAP Value"] < 0].head(2)

c_pos, c_neg = st.columns(2)
with c_pos:
    st.error("**Top Risk Drivers (Pushing towards Attrition):**")
    for _, row in top_pos.iterrows():
        st.write(
            "- "
            + get_business_interpretation(
                row["Feature"], row["SHAP Value"], row["Value"]
            )
        )

with c_neg:
    st.success("**Top Retention Drivers (Pushing towards Staying):**")
    for _, row in top_neg.iterrows():
        st.write(
            "- "
            + get_business_interpretation(
                row["Feature"], row["SHAP Value"], row["Value"]
            )
        )

st.divider()

# ==========================================
# EXPORT
# ==========================================
st.subheader("📥 Export SHAP Data")
c_exp1, c_exp2 = st.columns(2)

with c_exp1:
    csv_global = (
        pd.DataFrame(shap_values, columns=features).to_csv(index=False).encode("utf-8")
    )
    st.download_button(
        label="Download Global SHAP Matrix (CSV)",
        data=csv_global,
        file_name=f"employee_attrition_shap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

with c_exp2:
    csv_local = loc_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download Local Explanation for Emp {selected_emp_idx} (CSV)",
        data=csv_local,
        file_name=f"employee_attrition_shap_local_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

st.caption(
    "Generated by Employee Attrition Prediction System | Dynamic SHAP Explainability Module"
)

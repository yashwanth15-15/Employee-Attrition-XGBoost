import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, 
    roc_curve, precision_recall_curve, balanced_accuracy_score, 
    matthews_corrcoef, cohen_kappa_score
)

st.set_page_config(page_title="Model Performance", layout="wide")

st.title("🤖 Model Performance Dashboard")
st.markdown("""
This dashboard provides a comprehensive evaluation of the final deployment model (XGBoost) 
used in the Employee Attrition Prediction System. All metrics and visualizations are generated 
dynamically from the test dataset.
""")

# ==========================================
# DATA & MODEL LOADING
# ==========================================

@st.cache_resource
def load_model_artifacts():
    try:
        with open("models/final_xgboost_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("models/final_encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        with open("models/final_features.pkl", "rb") as f:
            features = pickle.load(f)
        return model, encoders, features
    except FileNotFoundError:
        return None, None, None

@st.cache_data
def load_and_preprocess_test_data(_encoders, features):
    try:
        df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    except FileNotFoundError:
        return None, None, None, None, None
        
    for col, enc in _encoders.items():
        if col in df.columns and col != "Attrition":
            df[col] = enc.transform(df[col].astype(str))
            
    df["Attrition"] = df["Attrition"].apply(lambda x: 1 if x == "Yes" else 0)
    
    X = df[features]
    y = df["Attrition"]
    
    # Must match original training split exactly
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, len(df)

model, encoders, features = load_model_artifacts()

if model is None:
    st.error("❌ Model artifacts missing! Ensure models exist in the 'models/' directory.")
    st.stop()

X_train, X_test, y_train, y_test, total_samples = load_and_preprocess_test_data(encoders, features)

if X_test is None:
    st.error("❌ Dataset missing! Ensure 'data/WA_Fn-UseC_-HR-Employee-Attrition.csv' exists.")
    st.stop()

# ==========================================
# METRICS COMPUTATION
# ==========================================
with st.spinner("Evaluating model on test dataset..."):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Advanced Metrics
    spec = tn / (tn + fp)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    report_dict = classification_report(y_test, y_pred, output_dict=True)

# ==========================================
# KPI CARDS
# ==========================================
st.markdown("### 📌 Model Summary")

kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5, kpi_c6, kpi_c7 = st.columns(7)
kpi_c1.metric("Model", "XGBoost")
kpi_c2.metric("Accuracy", f"{acc*100:.2f}%")
kpi_c3.metric("Precision", f"{prec*100:.2f}%")
kpi_c4.metric("Recall", f"{rec*100:.2f}%")
kpi_c5.metric("F1 Score", f"{f1:.4f}")
kpi_c6.metric("ROC-AUC", f"{roc_auc:.4f}")
kpi_c7.metric("Test Samples", f"{len(y_test)}")

st.divider()

# ==========================================
# VISUALIZATIONS
# ==========================================

col_v1, col_v2 = st.columns(2)

# 1. Confusion Matrix
with col_v1:
    st.subheader("1. Confusion Matrix")
    fig_cm = go.Figure(data=go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=['Predicted No', 'Predicted Yes'],
        y=['Actual No', 'Actual Yes'],
        hoverongaps=False,
        colorscale='Blues',
        text=[[f"True Negatives: {tn}", f"False Positives: {fp}"],
              [f"False Negatives: {fn}", f"True Positives: {tp}"]],
        texttemplate="%{text}",
        showscale=False
    ))
    fig_cm.update_layout(title="Confusion Matrix", width=400, height=400)
    st.plotly_chart(fig_cm, use_container_width=True)
    st.info(f"**Business Impact:** The model caught {tp} employees likely to leave (True Positives) but missed {fn} (False Negatives).")

# 2. Classification Report
with col_v2:
    st.subheader("2. Classification Report")
    df_report = pd.DataFrame(report_dict).transpose()
    df_report = df_report.round(4)
    st.dataframe(df_report.style.highlight_max(axis=0, subset=['precision', 'recall', 'f1-score'], color='lightgreen'), use_container_width=True)
    st.info("**Explanation:** Shows class-level breakdown. '1' represents Attrition (Yes), '0' represents Retention (No).")

st.divider()

col_v3, col_v4 = st.columns(2)

# 3. ROC Curve
with col_v3:
    st.subheader("3. ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"XGBoost (AUC={roc_auc:.3f})", mode='lines', fill='tozeroy', line=dict(color='blue')))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random Baseline", mode='lines', line=dict(dash='dash', color='grey')))
    fig_roc.update_layout(title=f"Receiver Operating Characteristic (AUC: {roc_auc:.3f})", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig_roc, use_container_width=True)
    st.info("**Interpretation:** A higher AUC (closer to 1.0) means the model is excellent at distinguishing between employees who stay and those who leave.")

# 4. Precision-Recall Curve
with col_v4:
    st.subheader("4. Precision-Recall Curve")
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=recall_curve, y=precision_curve, mode='lines', fill='tozeroy', line=dict(color='orange')))
    fig_pr.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(fig_pr, use_container_width=True)
    st.info("**Why it matters:** Because attrition is a minority class, the PR curve provides a better assessment of performance than ROC by focusing purely on the positive class.")

st.divider()

col_v5, col_v6 = st.columns(2)

# 5. Prediction Probability Distribution
with col_v5:
    st.subheader("5. Prediction Probabilities")
    df_prob = pd.DataFrame({'Probability': y_prob, 'Actual': ['Attrition' if y == 1 else 'No Attrition' for y in y_test]})
    fig_prob = px.histogram(df_prob, x='Probability', color='Actual', barmode='overlay', title="Prediction Probability Distribution", nbins=30)
    fig_prob.update_traces(opacity=0.75)
    st.plotly_chart(fig_prob, use_container_width=True)

# 6. Feature Importance Summary
with col_v6:
    st.subheader("6. Top 10 Important Features")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-10:]
        top_features = [features[i] for i in indices]
        top_importances = importances[indices]
        
        fig_feat = px.bar(x=top_importances, y=top_features, orientation='h', title="XGBoost Feature Importance")
        st.plotly_chart(fig_feat, use_container_width=True)
    else:
        st.warning("Model does not expose feature importances.")

st.divider()

# 7. Performance Metric Comparison
st.subheader("7. Overall Metrics Comparison")
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
    'Score': [acc, prec, rec, f1, roc_auc]
})
fig_metrics = px.bar(metrics_df, x='Metric', y='Score', text='Score', color='Metric', title="Performance Metric Comparison")
fig_metrics.update_traces(texttemplate='%{text:.3f}', textposition='outside')
fig_metrics.update_yaxes(range=[0, 1.1])
st.plotly_chart(fig_metrics, use_container_width=True)

st.divider()

# ==========================================
# MODEL INTERPRETATION
# ==========================================
st.subheader("🧠 Model Interpretation & Business Impact")

if rec > 0.8:
    strength_txt = f"Excellent Recall ({rec*100:.1f}%), meaning the model correctly identifies the vast majority of employees at risk."
    hr_action = "Maintain current risk threshold; the model provides a highly reliable early warning system."
elif rec > 0.5:
    strength_txt = f"Moderate Recall ({rec*100:.1f}%), catching more than half of the at-risk employees."
    hr_action = "Consider lowering the risk threshold slightly to capture more potential resignations before they happen."
else:
    strength_txt = f"Low Recall ({rec*100:.1f}%), missing a significant portion of resigning employees."
    hr_action = "Lower the risk threshold in the HR assistant to trigger interventions sooner for border-line cases."

if prec > 0.7:
    weakness_txt = f"High Precision ({prec*100:.1f}%), indicating very few false alarms."
else:
    weakness_txt = f"Lower Precision ({prec*100:.1f}%), meaning HR will investigate some employees (False Positives) who were not actually planning to leave."

st.success(f"**💪 Strengths:** {strength_txt} Additionally, a high ROC-AUC ({roc_auc:.3f}) shows strong separability.")
st.warning(f"**⚠️ Limitations:** {weakness_txt}")
st.info(f"**💼 Business Impact:** By catching {tp} out of {tp+fn} potential departures in this test set, HR can proactively intervene and potentially save ₹{tp * 50000:,} (assuming a conservative ₹50k replacement cost per employee).")
st.info(f"**💡 HR Recommendation:** {hr_action}")

st.divider()

col_adv, col_info = st.columns(2)

# ==========================================
# ADVANCED METRICS
# ==========================================
with col_adv:
    st.subheader("🔬 Advanced Metrics")
    adv_df = pd.DataFrame({
        "Metric": ["Specificity", "Sensitivity (Recall)", "Balanced Accuracy", "Matthews Corr. Coef. (MCC)", "Cohen's Kappa"],
        "Value": [f"{spec:.4f}", f"{rec:.4f}", f"{bal_acc:.4f}", f"{mcc:.4f}", f"{kappa:.4f}"]
    })
    st.dataframe(adv_df, hide_index=True, use_container_width=True)

# ==========================================
# MODEL INFORMATION
# ==========================================
with col_info:
    st.subheader("ℹ️ Model Information")
    hyperparams = model.get_params() if hasattr(model, 'get_params') else "N/A"
    n_estimators = hyperparams.get("n_estimators", "N/A") if isinstance(hyperparams, dict) else "N/A"
    max_depth = hyperparams.get("max_depth", "N/A") if isinstance(hyperparams, dict) else "N/A"
    
    info_df = pd.DataFrame({
        "Property": ["Algorithm", "Training Samples", "Testing Samples", "Total Features", "Target Classes", "Estimators", "Max Depth"],
        "Value": ["XGBoost Classifier", f"{len(y_train)}", f"{len(y_test)}", f"{len(features)}", "2 (Yes, No)", f"{n_estimators}", f"{max_depth}"]
    })
    st.dataframe(info_df, hide_index=True, use_container_width=True)

st.divider()

# ==========================================
# EXPORT
# ==========================================
st.subheader("📥 Export Reports")
c_exp1, c_exp2 = st.columns(2)

with c_exp1:
    csv_report = df_report.to_csv().encode('utf-8')
    st.download_button(
        label="Download Classification Report (CSV)",
        data=csv_report,
        file_name="classification_report.csv",
        mime="text/csv"
    )

with c_exp2:
    csv_metrics = metrics_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Performance Metrics (CSV)",
        data=csv_metrics,
        file_name="performance_metrics.csv",
        mime="text/csv"
    )

st.caption("Generated by Employee Attrition Prediction System | Dynamic Model Evaluation Module")
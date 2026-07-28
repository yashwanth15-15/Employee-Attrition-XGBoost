import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import streamlit as st

@st.cache_resource
def get_model_metrics():
    """
    Evaluates the XGBoost model on the test set dynamically.
    Returns: (accuracy_score, f1_score)
    """
    df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    
    with open("models/final_xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/final_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open("models/final_features.pkl", "rb") as f:
        features = pickle.load(f)
        
    for col, enc in encoders.items():
        if col in df.columns and col != "Attrition":
            df[col] = enc.transform(df[col].astype(str))
            
    df["Attrition"] = df["Attrition"].apply(lambda x: 1 if x == "Yes" else 0)
    
    X = df[features]
    y = df["Attrition"]
    
    # We must use exactly the same split config as training to get the true test score
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    return acc, f1

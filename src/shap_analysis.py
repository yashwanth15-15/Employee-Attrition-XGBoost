import pandas as pd
import shap

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

df = pd.read_csv(
    r"D:\Employee_Attrition_XGBoost_Project\data\WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

model = XGBClassifier(
    eval_metric="logloss",
    random_state=42
)

model.fit(X, y)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

shap.summary_plot(
    shap_values,
    X,
    show=True
)
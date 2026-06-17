import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

df = pd.read_csv(
    r"D:\Employee_Attrition_XGBoost_Project\data\WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop('Attrition', axis=1)
y = df['Attrition']

model = XGBClassifier(eval_metric='logloss')
model.fit(X, y)

with open(
    r"D:\Employee_Attrition_XGBoost_Project\models\xgboost_attrition.pkl",
    "wb"
) as f:
    pickle.dump(model, f)

print("Model Saved Successfully!")
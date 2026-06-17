import pandas as pd
import matplotlib.pyplot as plt
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

importance = model.feature_importances_

feature_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

print(feature_imp.head(10))

plt.figure(figsize=(10,6))
plt.barh(feature_imp['Feature'][:10], feature_imp['Importance'][:10])
plt.title("Top 10 Important Features")
plt.xlabel("Importance Score")
plt.show()

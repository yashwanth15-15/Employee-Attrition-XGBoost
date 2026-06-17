import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

# Load Dataset
df = pd.read_csv(
    r"D:\Employee_Attrition_XGBoost_Project\data\WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# Store encoders
encoders = {}

# Encode categorical columns
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features and Target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Save feature names
feature_names = list(X.columns)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train XGBoost
model = XGBClassifier(
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)

print(
    "Accuracy:",
    accuracy_score(y_test, pred)
)

# Save Model
with open(
    r"D:\Employee_Attrition_XGBoost_Project\models\xgboost_final.pkl",
    "wb"
) as f:
    pickle.dump(model, f)

# Save Encoders
with open(
    r"D:\Employee_Attrition_XGBoost_Project\models\encoders.pkl",
    "wb"
) as f:
    pickle.dump(encoders, f)

# Save Feature Names
with open(
    r"D:\Employee_Attrition_XGBoost_Project\models\feature_names.pkl",
    "wb"
) as f:
    pickle.dump(feature_names, f)

print("Model Saved")
print("Encoders Saved")
print("Feature Names Saved")
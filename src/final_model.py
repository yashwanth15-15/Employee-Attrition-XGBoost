import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Load IBM Dataset
df = pd.read_csv(
    r"D:\Employee_Attrition_XGBoost_Project\data\WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# Save encoders
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

# SMOTE
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

# Final Tuned XGBoost
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(
    X_train_smote,
    y_train_smote
)

# Prediction
pred = model.predict(X_test)

print("Accuracy:")
print(accuracy_score(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred))

# Save Final Model
with open(
    r"D:\Employee_Attrition_XGBoost_Project\models\final_xgboost_model.pkl",
    "wb"
) as f:
    pickle.dump(model, f)

# Save Encoders
with open(
    r"D:\Employee_Attrition_XGBoost_Project\models\final_encoders.pkl",
    "wb"
) as f:
    pickle.dump(encoders, f)

# Save Features
with open(
    r"D:\Employee_Attrition_XGBoost_Project\models\final_features.pkl",
    "wb"
) as f:
    pickle.dump(feature_names, f)

print("\nFinal Model Saved Successfully!")
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from xgboost import XGBClassifier

# Load Dataset
df = pd.read_csv(
    r"D:\Employee_Attrition_XGBoost_Project\data\employee_attrition_dataset_10000.csv"
)

print("Dataset Shape:")
print(df.shape)

# Encode categorical columns
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Target Column
target = "Attrition"

X = df.drop(target, axis=1)
y = df[target]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# XGBoost
model = XGBClassifier(
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred))
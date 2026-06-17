import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

df = pd.read_csv(
    r"D:\Employee_Attrition_XGBoost_Project\data\WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

model = XGBClassifier(
    eval_metric="logloss",
    random_state=42
)

model.fit(
    X_train_smote,
    y_train_smote
)

pred = model.predict(X_test)

print(classification_report(y_test, pred))
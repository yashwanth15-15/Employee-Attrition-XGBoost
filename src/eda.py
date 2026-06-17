import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
    r"D:\Employee_Attrition_XGBoost_Project\data\WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# Attrition Count
plt.figure(figsize=(6,4))
sns.countplot(x='Attrition', data=df)
plt.title("Employee Attrition Distribution")
plt.savefig("D:/Employee_Attrition_XGBoost_Project/screenshots/attrition_distribution.png")
plt.show()
# 1. Attrition Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Attrition', data=df)
plt.title("Employee Attrition Distribution")
plt.show()

# 2. Overtime vs Attrition
plt.figure(figsize=(6,4))
sns.countplot(x='OverTime', hue='Attrition', data=df)
plt.title("Overtime vs Attrition")
plt.show()

# 3. Job Satisfaction vs Attrition
plt.figure(figsize=(6,4))
sns.countplot(x='JobSatisfaction', hue='Attrition', data=df)
plt.title("Job Satisfaction vs Attrition")
plt.show()

# 4. Department vs Attrition
plt.figure(figsize=(8,5))
sns.countplot(x='Department', hue='Attrition', data=df)
plt.xticks(rotation=20)
plt.title("Department vs Attrition")
plt.show()

# 5. Monthly Income Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['MonthlyIncome'], bins=30)
plt.title("Monthly Income Distribution")
plt.show()
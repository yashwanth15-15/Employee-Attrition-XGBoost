```python
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

st.markdown("""
# 🎯 Employee Attrition Prediction System

### XGBoost Powered HR Analytics Dashboard
""")

uploaded_file = st.file_uploader(
    "Upload Employee CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        # Load CSV
        df = pd.read_csv(uploaded_file)
        original_df = df.copy()

        st.subheader("Uploaded Data")
        st.dataframe(df.head())

        # Load Model
        with open(
            "models/final_xgboost_model.pkl",
            "rb"
        ) as f:
            model = pickle.load(f)

        # Load Encoders
        with open(
            "models/final_encoders.pkl",
            "rb"
        ) as f:
            encoders = pickle.load(f)

        # Load Feature Names
        with open(
            "models/final_features.pkl",
            "rb"
        ) as f:
            feature_names = pickle.load(f)

        # Validate Columns
        missing_cols = [
            col for col in feature_names
            if col not in df.columns
        ]

        if missing_cols:
            st.error(
                f"Missing Columns: {missing_cols}"
            )
            st.stop()

        # Encode Categorical Columns
        for col, encoder in encoders.items():

            if col == "Attrition":
                continue

            if col in df.columns:
                df[col] = encoder.transform(df[col])

        # Select Features
        X = df[feature_names]

        # Prediction
        probabilities = model.predict_proba(X)[:, 1]

        df["Attrition_Risk"] = (
            probabilities * 100
        ).round(2)

        # Risk Level
        def risk_level(x):

            if x >= 70:
                return "High"

            elif x >= 40:
                return "Medium"

            else:
                return "Low"

        df["Risk_Level"] = df[
            "Attrition_Risk"
        ].apply(risk_level)

        # Sort by Risk
        df = df.sort_values(
            by="Attrition_Risk",
            ascending=False
        )

        # Employee Ranking
        st.subheader(
            "Employee Risk Ranking"
        )

        st.dataframe(df)

        # Risk Summary
        high_risk = len(
            df[df["Risk_Level"] == "High"]
        )

        medium_risk = len(
            df[df["Risk_Level"] == "Medium"]
        )

        low_risk = len(
            df[df["Risk_Level"] == "Low"]
        )

        st.subheader("📊 Risk Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "🔴 High Risk",
                high_risk
            )

        with col2:
            st.metric(
                "🟡 Medium Risk",
                medium_risk
            )

        with col3:
            st.metric(
                "🟢 Low Risk",
                low_risk
            )

        # Pie Chart
        st.subheader(
            "🥧 Employee Risk Distribution"
        )

        risk_df = pd.DataFrame({
            "Category": [
                "High",
                "Medium",
                "Low"
            ],
            "Count": [
                high_risk,
                medium_risk,
                low_risk
            ]
        })

        fig = px.pie(
            risk_df,
            names="Category",
            values="Count",
            hole=0.4
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Department Analysis
        if "Department" in original_df.columns:

            st.subheader(
                "🏢 Department-wise Attrition Risk"
            )

            dept_analysis = (
                pd.concat(
                    [
                        original_df["Department"],
                        df["Attrition_Risk"]
                    ],
                    axis=1
                )
                .groupby("Department")
                ["Attrition_Risk"]
                .mean()
                .reset_index()
                .sort_values(
                    by="Attrition_Risk",
                    ascending=False
                )
            )

            st.dataframe(
                dept_analysis
            )

            fig = px.bar(
                dept_analysis,
                x="Department",
                y="Attrition_Risk",
                title="Department-wise Attrition Risk",
                text_auto=".2f"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            highest_dept = dept_analysis.iloc[0]

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Highest Risk Department",
                    highest_dept["Department"]
                )

            with col2:
                st.metric(
                    "Average Risk",
                    f"{highest_dept['Attrition_Risk']:.2f}%"
                )

        # Search Employee
        st.subheader(
            "🔍 Search Employee"
        )

        search_id = st.text_input(
            "Enter Employee Number"
        )

        if search_id:

            result = df[
                df["EmployeeNumber"]
                .astype(str)
                == search_id
            ]

            if not result.empty:
                st.dataframe(result)
            else:
                st.warning(
                    "Employee not found"
                )

        # Top 5
        st.subheader(
            "🏆 Top 5 Employees Most Likely To Leave"
        )

        top5 = df.head(5)

        st.dataframe(
            top5[
                [
                    "EmployeeNumber",
                    "Attrition_Risk",
                    "Risk_Level"
                ]
            ]
        )

        # Statistics
        st.subheader(
            "📈 Quick Statistics"
        )

        st.write(
            f"Total Employees Analysed: {len(df)}"
        )

        st.write(
            f"Average Attrition Risk: {df['Attrition_Risk'].mean():.2f}%"
        )

        st.write(
            f"Maximum Attrition Risk: {df['Attrition_Risk'].max():.2f}%"
        )

        st.write(
            f"Minimum Attrition Risk: {df['Attrition_Risk'].min():.2f}%"
        )

        # Highest Risk Employee
        highest = df.iloc[0]

        st.markdown("---")

        st.subheader(
            "🚨 Highest Risk Employee"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Employee ID",
                highest["EmployeeNumber"]
            )

        with col2:
            st.metric(
                "Risk %",
                f"{highest['Attrition_Risk']:.2f}%"
            )

        with col3:
            st.metric(
                "Risk Level",
                highest["Risk_Level"]
            )

        # HR Insights
        st.subheader(
            "💡 HR Insights"
        )

        if high_risk > 0:

            st.warning(
                f"{high_risk} employees are at high attrition risk."
            )

            st.write("""
• Conduct retention discussions

• Review compensation

• Improve work-life balance

• Provide career growth opportunities

• Reduce excessive overtime
""")

        else:

            st.success(
                "No high-risk employees detected."
            )

        # Download Results
        csv = df.to_csv(
            index=False
        )

        st.download_button(
            label="⬇ Download Prediction Report",
            data=csv,
            file_name="attrition_predictions.csv",
            mime="text/csv"
        )

        st.success(
            "Prediction completed successfully!"
        )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )

st.markdown("---")

st.caption(
    "Employee Attrition Prediction System | XGBoost Machine Learning Model | B.Tech Major Project"
)
```

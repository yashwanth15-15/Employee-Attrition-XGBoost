import pickle
import pandas as pd
import shap
import numpy as np
from typing import Dict, Tuple, Any
from api.config import settings
from api.core.exceptions import ModelLoadError
from api.core.logger import logger
from App.helpers.risk_calculator import calculate_risk

class MLService:
    def __init__(self):
        self.model = None
        self.encoders = None
        self.feature_names = None
        self.explainer = None
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            with open(settings.XGBOOST_MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            with open(settings.ENCODERS_PATH, "rb") as f:
                self.encoders = pickle.load(f)
            with open(settings.FEATURES_PATH, "rb") as f:
                self.feature_names = pickle.load(f)
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("Successfully loaded ML artifacts.")
        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")
            raise ModelLoadError(str(e))

    def _preprocess_features(self, features_dict: Dict[str, Any]) -> pd.DataFrame:
        df = pd.DataFrame([features_dict])
        
        # Mapping frontend names to model internal names
        rename_map = {
            "Monthly Income": "MonthlyIncome",
            "Total Working Years": "TotalWorkingYears",
            "Years At Company": "YearsAtCompany",
            "Years In Current Role": "YearsInCurrentRole",
            "Years Since Last Promotion": "YearsSinceLastPromotion",
            "Marital Status": "MaritalStatus",
            "Business Travel": "BusinessTravel",
            "Work Life Balance": "WorkLifeBalance",
            "Job Satisfaction": "JobSatisfaction",
            "Environment Satisfaction": "EnvironmentSatisfaction",
            "Stock Option Level": "StockOptionLevel",
            "Training Times Last Year": "TrainingTimesLastYear"
        }
        df.rename(columns=rename_map, inplace=True)
        
        # Missing defaults
        if 'PerformanceRating' not in df.columns:
            df['PerformanceRating'] = features_dict.get('Performance Rating', 3)
        if 'YearsWithCurrManager' not in df.columns:
            df['YearsWithCurrManager'] = features_dict.get('Years In Current Role', 0)
        if 'JobInvolvement' not in df.columns:
            df['JobInvolvement'] = 3
        if 'Education' not in df.columns:
            df['Education'] = 3
        if 'NumCompaniesWorked' not in df.columns:
            df['NumCompaniesWorked'] = 1
        if 'RelationshipSatisfaction' not in df.columns:
            df['RelationshipSatisfaction'] = 3
        if 'JobLevel' not in df.columns:
            df['JobLevel'] = 1
        if 'PercentSalaryHike' not in df.columns:
            df['PercentSalaryHike'] = 11

        # Ensure all required features are present
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0

        # Encode categorical variables
        for col, encoder in self.encoders.items():
            if col == "Attrition": continue
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except Exception:
                    pass
                    
        return df[self.feature_names]

    def predict(self, features_dict: Dict[str, Any]) -> Tuple[float, str, Dict[str, float]]:
        X = self._preprocess_features(features_dict)
        prob = float(self.model.predict_proba(X)[0][1])
        risk_category = calculate_risk(prob)
        
        # SHAP calculation
        shap_vals = self.explainer.shap_values(X)
        contributions = shap_vals[0] * 100
        
        # Extract top 5 shap values
        abs_s = np.abs(contributions)
        valid_idx = [i for i in range(len(abs_s)) if abs_s[i] > 1.0]
        top_idx = sorted(valid_idx, key=lambda i: abs_s[i], reverse=True)[:5]
        
        top_shap_dict = {self.feature_names[i]: float(contributions[i]) for i in top_idx}
        
        return prob, risk_category, top_shap_dict

ml_service = MLService()

import pickle
import time
from typing import Any, Dict, List, Set, Tuple, TypedDict

import numpy as np
import pandas as pd
import shap

from api.config import settings
from api.core.constants import (
    DEFAULT_EDUCATION,
    DEFAULT_JOB_INVOLVEMENT,
    DEFAULT_JOB_LEVEL,
    DEFAULT_NUM_COMPANIES_WORKED,
    DEFAULT_PERFORMANCE_RATING,
    DEFAULT_RELATIONSHIP_SATISFACTION,
    DEFAULT_SALARY_HIKE,
    MAX_TOP_FEATURES,
)
from api.core.exceptions import (ModelLoadError, PredictionError,
                                 PreprocessingError, SHAPGenerationError)
from api.core.logger import logger
from api.utils.risk_calculator import calculate_risk



class RiskDriver(TypedDict):
    feature: str
    contribution: float
    impact: str
    shap_value: float


class MLService:
    """
    Machine Learning Service for Employee Attrition Prediction.
    Provides prediction logic, data preprocessing, and SHAP explainability.
    """

    # Immutable feature mapping to avoid recreating it per request
    RENAME_MAP = {
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
        "Training Times Last Year": "TrainingTimesLastYear",
    }

    def __init__(self):
        """Initializes the ML service and loads cached models."""
        self.model = None
        self.encoders = None
        self.feature_names: List[str] = []
        self.explainer = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads the XGBoost model, encoders, and SHAP explainer from disk."""
        logger.info("Loading ML artifacts...")
        try:
            with open(settings.XGBOOST_MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            with open(settings.ENCODERS_PATH, "rb") as f:
                self.encoders = pickle.load(f)
            with open(settings.FEATURES_PATH, "rb") as f:
                self.feature_names = pickle.load(f)

            self.explainer = shap.TreeExplainer(self.model)
            logger.info("Successfully loaded ML artifacts and SHAP explainer.")
        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")
            raise ModelLoadError(str(e))

    def _validate_features(self, df: pd.DataFrame, incoming_keys: Set[str]) -> None:
        """Validates incoming features against the expected model features."""
        expected_keys = set(self.feature_names)

        # Check for unexpected features
        unexpected = incoming_keys - expected_keys
        if unexpected:
            logger.warning(f"Unexpected features received and ignored: {unexpected}")

        # Check for missing required features
        missing = expected_keys - incoming_keys - set(self.RENAME_MAP.values())
        if missing:
            logger.warning(
                f"Missing expected features (will be defaulted to 0): {missing}"
            )

    def _apply_defaults(
        self, features_dict: Dict[str, Any], df: pd.DataFrame
    ) -> pd.DataFrame:
        """Applies business-logic defaults for missing values."""
        if "PerformanceRating" not in df.columns:
            df["PerformanceRating"] = features_dict.get(
                "Performance Rating", DEFAULT_PERFORMANCE_RATING
            )
        if "YearsWithCurrManager" not in df.columns:
            df["YearsWithCurrManager"] = features_dict.get("Years In Current Role", 0)
        if "JobInvolvement" not in df.columns:
            df["JobInvolvement"] = DEFAULT_JOB_INVOLVEMENT
        if "Education" not in df.columns:
            df["Education"] = DEFAULT_EDUCATION
        if "NumCompaniesWorked" not in df.columns:
            df["NumCompaniesWorked"] = DEFAULT_NUM_COMPANIES_WORKED
        if "RelationshipSatisfaction" not in df.columns:
            df["RelationshipSatisfaction"] = DEFAULT_RELATIONSHIP_SATISFACTION
        if "JobLevel" not in df.columns:
            df["JobLevel"] = DEFAULT_JOB_LEVEL
        if "PercentSalaryHike" not in df.columns:
            df["PercentSalaryHike"] = DEFAULT_SALARY_HIKE

        return df

    def _encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encodes categorical features using pre-loaded LabelEncoders."""
        for col, encoder in self.encoders.items():
            if col == "Attrition":
                continue
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except Exception as e:
                    logger.warning("Encoding failed for column '%s': %s", col, str(e))
        return df

    def _prepare_dataframe(self, features_dict: Dict[str, Any]) -> pd.DataFrame:
        """Converts dict to DataFrame and standardizes columns for prediction."""
        try:
            df = pd.DataFrame([features_dict])
            df.rename(columns=self.RENAME_MAP, inplace=True)

            self._validate_features(df, set(df.columns))
            df = self._apply_defaults(features_dict, df)

            # Ensure all required features exist
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0

            df = self._encode_features(df)

            # Return ordered dataframe based on exact training columns
            return df[self.feature_names]
        except PreprocessingError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during preprocessing: {str(e)}")
            raise PreprocessingError(f"Unexpected error during preprocessing: {str(e)}")

    def _generate_prediction(self, X: pd.DataFrame) -> float:
        """Executes the machine learning prediction to compute attrition probability."""
        try:
            start_time = time.time()
            prob = float(self.model.predict_proba(X)[0][1])
            duration = time.time() - start_time
            logger.info(
                f"Prediction generated successfully in {duration:.4f}s. Probability: {prob:.4f}"
            )
            return prob
        except Exception as e:
            logger.error(f"Prediction logic failed: {str(e)}")
            raise PredictionError(f"Failed to generate prediction probability.")

    def _generate_shap(self, X: pd.DataFrame) -> List[RiskDriver]:
        """Generates SHAP explainability insights to identify risk drivers."""
        try:
            start_time = time.time()
            shap_vals = self.explainer.shap_values(X)[0]
            abs_shap = np.abs(shap_vals)
            total = np.sum(abs_shap)

            if total == 0:
                total = 1.0

            top_risk_drivers: List[RiskDriver] = []
            valid_idx = [i for i in range(len(abs_shap)) if abs_shap[i] > 0.0]
            top_idx = sorted(valid_idx, key=lambda i: abs_shap[i], reverse=True)[
                :MAX_TOP_FEATURES
            ]

            for i in top_idx:
                shap_value = float(shap_vals[i])
                contribution = (abs(shap_value) / total) * 100
                impact = "increase" if shap_value > 0 else "decrease"

                top_risk_drivers.append(
                    {
                        "feature": self.feature_names[i],
                        "contribution": contribution,
                        "impact": impact,
                        "shap_value": shap_value,
                    }
                )

            duration = time.time() - start_time
            logger.info(f"SHAP generation completed in {duration:.4f}s.")
            return top_risk_drivers
        except Exception as e:
            logger.error(f"SHAP explanation generation failed: {str(e)}")
            raise SHAPGenerationError(f"Failed to generate SHAP explanations.")

    def predict(
        self, features_dict: Dict[str, Any]
    ) -> Tuple[float, str, List[RiskDriver]]:
        """
        Main entry point for generating predictions.
        Prepares data, generates probabilities, calculates risk, and provides SHAP explanations.
        """
        logger.info("Prediction request received.")
        X = self._prepare_dataframe(features_dict)

        prob = self._generate_prediction(X)
        risk_category = calculate_risk(prob)
        top_risk_drivers = self._generate_shap(X)

        logger.info(f"Prediction completed. Risk: {risk_category}")
        return prob, risk_category, top_risk_drivers


# Instantiate singleton ML Service
ml_service = MLService()

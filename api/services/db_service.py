import os
import sys
from typing import Any, Dict, List

import pandas as pd

from api.core.exceptions import DatabaseError
from api.core.logger import logger

from sqlalchemy.orm import Session
from api.database import crud


class DBService:
    @staticmethod
    def _map_features_to_db_record(
        emp_id: str,
        features: Dict[str, Any],
        prob: float,
        risk: str,
        shap_dict: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Maps API feature names to database schema column names.
        
        Args:
            emp_id (str): Generated employee ID.
            features (Dict[str, Any]): The employee features.
            prob (float): Predicted probability of attrition.
            risk (str): Risk category.
            shap_dict (Dict[str, float]): SHAP explanations.

        Returns:
            Dict[str, Any]: Formatted record ready for SQLite.
        """
        age = features.get("Age", 30)
        gender = features.get("Gender", "Unknown")
        department = features.get("Department", "Unknown")
        salary = features.get("Monthly Income", 0)

        # Map satisfaction to 1-4 scale
        js = features.get("Job Satisfaction", 3)
        es = features.get("Environment Satisfaction", 3)
        wlb = features.get("Work Life Balance", 3)

        factors = ", ".join([f"{k}: {v:.1f}%" for k, v in shap_dict.items()])

        return {
            "prediction_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "employee_id": emp_id,
            "employee_name": f"Employee_{emp_id}",
            "age": age,
            "gender": gender,
            "department": department,
            "marital_status": features.get("Marital Status", "Single"),
            "monthly_income": salary,
            "years_at_company": features.get("Years At Company", 0),
            "job_satisfaction": js,
            "work_life_balance": wlb,
            "overtime": features.get("OverTime", "No"),
            "prediction_probability": prob,
            "risk_category": risk,
            "health_score": int((1 - prob) * 100),
            "replacement_cost": salary * 12,
            "shap_summary": factors,
        }

    @staticmethod
    def save_prediction_record(
        db: Session,
        emp_id: str,
        features: Dict[str, Any],
        prob: float,
        risk: str,
        shap_dict: Dict[str, float],
    ) -> int:
        """
        Formats features and saves the prediction record to the database.

        Args:
            emp_id (str): The unique identifier for the employee.
            features (Dict[str, Any]): Original features submitted.
            prob (float): Attrition probability score.
            risk (str): Categorized risk level.
            shap_dict (Dict[str, float]): Feature importance dictionary.

        Returns:
            int: The inserted record ID.
            
        Raises:
            DatabaseError: If inserting into SQLite fails.
        """
        try:
            record = DBService._map_features_to_db_record(
                emp_id, features, prob, risk, shap_dict
            )
            return crud.create_prediction(db, record)
        except Exception as e:
            logger.error(f"Failed to save prediction: {e}")
            raise DatabaseError(f"Database save error: {str(e)}")

    @staticmethod
    def get_all_predictions(db: Session) -> List[Dict[str, Any]]:
        try:
            records = crud.get_all_predictions(db)
            return [{c.name: getattr(r, c.name) for c in r.__table__.columns} for r in records]
        except Exception as e:
            logger.error(f"Failed to get predictions: {e}")
            raise DatabaseError(f"Database read error: {str(e)}")

    @staticmethod
    def get_user_by_username(db: Session, username: str) -> Dict[str, Any]:
        try:
            user = crud.get_user_by_username(db, username)
            if user:
                return {c.name: getattr(user, c.name) for c in user.__table__.columns}
            return None
        except Exception as e:
            logger.error(f"Failed to get user {username}: {e}")
            raise DatabaseError(f"Database read error: {str(e)}")

    @staticmethod
    def create_user(db: Session, user: Dict[str, Any]) -> None:
        try:
            crud.create_user(db, user)
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            raise DatabaseError(f"Database save error: {str(e)}")

db_service = DBService()

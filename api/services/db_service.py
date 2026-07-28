import sys
import os
from typing import Dict, Any, List
import pandas as pd
from api.core.logger import logger
from api.core.exceptions import DatabaseError

# We temporarily append App dir to path to reuse the exact database file
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from App.database import get_predictions, add_prediction

class DBService:
    @staticmethod
    def save_prediction_record(emp_id: str, features: Dict[str, Any], prob: float, risk: str, shap_dict: Dict[str, float]) -> int:
        try:
            # We must map our features dict back to kwargs expected by save_prediction
            # save_prediction expects: emp_id, age, gender, department, role, salary, satisfaction, prob, risk, top_factors
            
            # Since the frontend might pass different names, we use defaults
            age = features.get('Age', 30)
            gender = features.get('Gender', 'Unknown')
            department = features.get('Department', 'Unknown')
            role = features.get('JobRole', 'Unknown')
            salary = features.get('Monthly Income', 0)
            
            # Map satisfaction to 1-4 scale, taking average if needed
            js = features.get('Job Satisfaction', 3)
            es = features.get('Environment Satisfaction', 3)
            wlb = features.get('Work Life Balance', 3)
            satisfaction = round((js + es + wlb) / 3, 1)
            
            factors = ", ".join([f"{k}: {v:.1f}%" for k, v in shap_dict.items()])
            
            # For phase 4 requirements, we should just call add_prediction
            record = {
                'prediction_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'employee_name': f"Employee_{emp_id}",
                'age': age,
                'gender': gender,
                'department': department,
                'marital_status': features.get('Marital Status', 'Single'),
                'monthly_income': salary,
                'years_at_company': features.get('Years At Company', 0),
                'job_satisfaction': js,
                'work_life_balance': wlb,
                'overtime': features.get('OverTime', 'No'),
                'prediction_probability': prob,
                'risk_category': risk,
                'health_score': int((1 - prob) * 100),
                'replacement_cost': salary * 12
            }
            return add_prediction(record)
        except Exception as e:
            logger.error(f"Failed to save prediction: {e}")
            raise DatabaseError(f"Database save error: {str(e)}")

    @staticmethod
    def get_all_predictions() -> List[Dict[str, Any]]:
        try:
            df = get_predictions()
            if df.empty:
                return []
            return df.to_dict(orient="records")
        except Exception as e:
            logger.error(f"Failed to get predictions: {e}")
            raise DatabaseError(f"Database read error: {str(e)}")

db_service = DBService()

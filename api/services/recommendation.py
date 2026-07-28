from typing import Dict, Any
from App.helpers.hr_recommendation_engine import generate_hr_recommendation
from api.models.schemas import HRRecommendations

class RecommendationService:
    @staticmethod
    def generate(features: Dict[str, Any], prob: float, risk: str) -> HRRecommendations:
        # generate_hr_recommendation expects a standard dictionary
        standard_dict = {
            "Age": features.get("Age", 30),
            "Gender": features.get("Gender", "Male"),
            "Department": features.get("Department", "Sales"),
            "Monthly Income": features.get("Monthly Income", 5000),
            "Marital Status": features.get("Marital Status", "Single"),
            "OverTime": features.get("OverTime", "No"),
            "Years At Company": features.get("Years At Company", 0),
            "Total Working Years": features.get("Total Working Years", 0),
            "Job Satisfaction": features.get("Job Satisfaction", 3),
            "Environment Satisfaction": features.get("Environment Satisfaction", 3),
            "Work Life Balance": features.get("Work Life Balance", 3),
            "Years Since Last Promotion": features.get("Years Since Last Promotion", 0),
            "Training Times Last Year": features.get("Training Times Last Year", 2),
            "Business Travel": features.get("Business Travel", "Non-Travel"),
            "Distance From Home": features.get("Distance From Home", 5),
            "Performance Rating": features.get("Performance Rating", 3),
            "Stock Option Level": features.get("Stock Option Level", 0),
            "Years In Current Role": features.get("Years In Current Role", 0),
        }
        rec_dict = generate_hr_recommendation(standard_dict, prob, risk)
        
        return HRRecommendations(**rec_dict)

recommendation_service = RecommendationService()

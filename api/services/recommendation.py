from typing import Any, Dict

from api.core.constants import DEFAULT_RECOMMENDATION_FEATURES
from api.models.schemas import HRRecommendations
from app.helpers.hr_recommendation_engine import generate_hr_recommendation


class RecommendationService:
    @staticmethod
    def generate(features: Dict[str, Any], prob: float, risk: str) -> HRRecommendations:
        # generate_hr_recommendation expects a standard dictionary
        # Merge the incoming features over the default base dictionary
        standard_dict = {**DEFAULT_RECOMMENDATION_FEATURES, **features}
        
        rec_dict = generate_hr_recommendation(standard_dict, prob, risk)

        return HRRecommendations(**rec_dict)


recommendation_service = RecommendationService()

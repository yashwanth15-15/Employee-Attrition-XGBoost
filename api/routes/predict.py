from fastapi import APIRouter, HTTPException
from api.models.schemas import EmployeeFeatures, PredictionResponse
from api.services.ml_service import ml_service
from api.services.recommendation import recommendation_service
from api.services.db_service import db_service
import uuid

router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.post("", response_model=PredictionResponse, summary="Predict Attrition Risk", description="Submit employee features to receive the attrition probability, SHAP analysis, and HR recommendations.")
async def predict_attrition(employee: EmployeeFeatures):
    # Convert Pydantic model to dictionary using aliases
    features_dict = employee.model_dump(by_alias=True)
    
    # 1. Run ML Model
    prob, risk_category, shap_dict = ml_service.predict(features_dict)
    
    # 2. Generate Recommendations
    recommendations = recommendation_service.generate(features_dict, prob, risk_category)
    
    # 3. Save to database
    emp_id = str(uuid.uuid4())[:8]
    pred_id = db_service.save_prediction_record(
        emp_id=emp_id,
        features=features_dict,
        prob=prob,
        risk=risk_category,
        shap_dict=shap_dict
    )
    
    return PredictionResponse(
        probability=prob,
        risk_category=risk_category,
        shap_values=shap_dict,
        recommendations=recommendations,
        prediction_id=pred_id
    )

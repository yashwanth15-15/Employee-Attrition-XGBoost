import uuid

from fastapi import APIRouter

from api.core.exceptions import APIException
from api.core.logger import logger
from api.models.schemas import EmployeeFeatures, PredictionResponse
from api.services.db_service import db_service
from api.services.ml_service import ml_service
from api.services.recommendation import recommendation_service

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post(
    "",
    response_model=PredictionResponse,
    summary="Predict Attrition Risk",
    description="Submit employee features to receive the attrition probability, SHAP analysis, and HR recommendations.",
)
async def predict_attrition(employee: EmployeeFeatures) -> PredictionResponse:
    """
    Endpoint to predict employee attrition risk, generate SHAP explanations,
    and formulate HR recommendations.
    """
    logger.info("Request received for /predict")
    try:
        # Convert Pydantic model to dictionary using aliases
        features_dict = employee.model_dump(by_alias=True)

        # 1. Run ML Model
        prob, risk_category, top_risk_drivers = ml_service.predict(features_dict)

        # 2. Generate Recommendations
        recommendations = recommendation_service.generate(
            features_dict, prob, risk_category
        )

        # Convert top_risk_drivers to shap_dict for DB backward compatibility
        shap_dict = {d["feature"]: d["shap_value"] for d in top_risk_drivers}

        # 3. Save to database
        emp_id = str(uuid.uuid4())[:8]
        pred_id = db_service.save_prediction_record(
            emp_id=emp_id,
            features=features_dict,
            prob=prob,
            risk=risk_category,
            shap_dict=shap_dict,
        )
        
        logger.info(f"Successfully processed /predict for employee_id: {emp_id}")

        return PredictionResponse(
            probability=prob,
            risk_category=risk_category,
            top_risk_drivers=top_risk_drivers,
            recommendations=recommendations,
            prediction_id=pred_id,
        )
    except APIException as e:
        logger.error(f"APIException in /predict: {e.message}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /predict: {str(e)}")
        raise APIException(f"Failed to process prediction request: {str(e)}", status_code=500)

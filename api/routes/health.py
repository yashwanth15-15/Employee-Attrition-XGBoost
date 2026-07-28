from fastapi import APIRouter
from pydantic import BaseModel
import fastapi
from datetime import datetime
from api.services.ml_service import ml_service
from api.services.db_service import db_service
from api.services.recommendation import recommendation_service

router = APIRouter(prefix="/health", tags=["System"])

class HealthResponse(BaseModel):
    status: str
    message: str
    fastapi_version: str
    model_loaded: bool
    database_connected: bool
    shap_available: bool
    recommendation_engine_available: bool
    timestamp: str

@router.get("", response_model=HealthResponse, summary="Check API health and dependency status")
async def health_check():
    # Check model
    model_loaded = ml_service.model is not None
    shap_loaded = ml_service.explainer is not None
    
    # Check database
    db_connected = False
    try:
        db_service.get_all_predictions()
        db_connected = True
    except Exception:
        pass
        
    return HealthResponse(
        status="ok",
        message="FastAPI backend is running successfully.",
        fastapi_version=fastapi.__version__,
        model_loaded=model_loaded,
        database_connected=db_connected,
        shap_available=shap_loaded,
        recommendation_engine_available=True,  # Engine is a static module
        timestamp=datetime.now().isoformat()
    )

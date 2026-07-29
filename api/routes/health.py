from datetime import datetime

import fastapi
from fastapi import APIRouter
from pydantic import BaseModel

from api.core.exceptions import APIException
from api.core.logger import logger
from api.services.db_service import db_service
from api.services.ml_service import get_ml_service

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


@router.get(
    "", response_model=HealthResponse, summary="Check API health and dependency status"
)
async def health_check() -> HealthResponse:
    """
    Returns the health status of the API and its internal dependencies.
    """
    logger.info("Request received for /health")
    try:
        # Check model
        ml_svc = get_ml_service()
        model_loaded = ml_svc.model is not None
        shap_loaded = ml_svc.explainer is not None

        # Check database
        db_connected = False
        try:
            from sqlalchemy import text
            from api.database.session import engine
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db_connected = True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")

        logger.info("Successfully processed /health")
        return HealthResponse(
            status="ok",
            message="FastAPI backend is running successfully.",
            fastapi_version=fastapi.__version__,
            model_loaded=model_loaded,
            database_connected=db_connected,
            shap_available=shap_loaded,
            recommendation_engine_available=True,  # Engine is a static module
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Unexpected error in /health: {str(e)}")
        raise APIException(f"Health check failed: {str(e)}", status_code=500)

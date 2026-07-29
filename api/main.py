import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.config import settings
from api.core.error_handlers import add_exception_handlers
from api.core.logger import logger
from api.routes import analytics, auth, health, predict, reports, simulate


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up API - Validating artifacts...")

    # Validate DB Path
    if not settings.DATABASE_PATH.parent.exists():
        logger.error(
            f"Database directory {settings.DATABASE_PATH.parent} does not exist."
        )
        raise RuntimeError("Database directory missing.")

    # Validate Model Paths
    for path, name in [
        (settings.XGBOOST_MODEL_PATH, "Model"),
        (settings.ENCODERS_PATH, "Encoders"),
        (settings.FEATURES_PATH, "Features"),
    ]:
        if not path.exists():
            logger.error(f"{name} file not found at {path}")
            raise RuntimeError(f"Missing artifact: {name}")

    # Import ML service and explicitly validate
    from api.services.ml_service import ml_service

    if not hasattr(ml_service.model, "predict_proba"):
        logger.error("Loaded model does not support predict_proba.")
        raise RuntimeError("Invalid model format.")

    if ml_service.explainer is None:
        logger.error("SHAP explainer failed to initialize.")
        raise RuntimeError("Invalid SHAP configuration.")
        
    # Setup Auth defaults
    from api.auth.auth_service import auth_service
    auth_service.create_default_admin()

    logger.info("Startup validation passed successfully.")
    yield
    logger.info("Shutting down API...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Enterprise API for Employee Attrition Prediction System",
    lifespan=lifespan,
)

# CORS Configuration
if hasattr(settings, "ALLOWED_ORIGINS") and settings.ALLOWED_ORIGINS:
    origins = [origin.strip() for origin in settings.ALLOWED_ORIGINS.split(",")]
else:
    origins = ["http://localhost:8501", "http://127.0.0.1:8501"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Incoming request: {request.method} {request.url.path}")

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(
        f"Response status: {response.status_code} - Execution time: {process_time:.4f}s"
    )
    return response


# Exception handlers
add_exception_handlers(app)

# Routes
app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(health.router, prefix=settings.API_V1_STR)
app.include_router(predict.router, prefix=settings.API_V1_STR)
app.include_router(simulate.router, prefix=settings.API_V1_STR)
app.include_router(analytics.router, prefix=settings.API_V1_STR)
app.include_router(reports.router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Root"])
async def root_endpoint():
    return {
        "application": "Employee Attrition Prediction API",
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": f"{settings.API_V1_STR}/health",
    }

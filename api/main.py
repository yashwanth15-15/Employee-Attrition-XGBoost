import sys
import time
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.config import settings
from api.core.error_handlers import add_exception_handlers
from api.core.logger import logger
from api.routes import analytics, auth, health, predict, reports, simulate


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_start = time.time()
    logger.info("=" * 60)
    logger.info("STARTUP BEGIN — Employee Attrition API")
    logger.info(f"  Python: {sys.version}")
    logger.info(f"  BASE_DIR: {settings.BASE_DIR}")
    logger.info(f"  DATABASE_PATH: {settings.DATABASE_PATH}")
    logger.info(f"  DATABASE_URL: {settings.DATABASE_URL}")
    logger.info(f"  MODELS_DIR: {settings.MODELS_DIR}")
    logger.info("=" * 60)
    sys.stdout.flush()

    # --- Step 1: Validate DB Path ---
    try:
        step_start = time.time()
        logger.info("[1/5] Validating database path...")
        if not settings.DATABASE_PATH.parent.exists():
            logger.error(
                f"Database directory {settings.DATABASE_PATH.parent} does not exist."
            )
            raise RuntimeError("Database directory missing.")
        logger.info(f"[1/5] Database path OK ({time.time() - step_start:.2f}s)")
    except Exception:
        logger.error(f"[1/5] FAILED:\n{traceback.format_exc()}")
        raise

    # --- Step 2: Validate Model Paths ---
    try:
        step_start = time.time()
        logger.info("[2/5] Validating model artifact paths...")
        for path, name in [
            (settings.XGBOOST_MODEL_PATH, "Model"),
            (settings.ENCODERS_PATH, "Encoders"),
            (settings.FEATURES_PATH, "Features"),
        ]:
            if not path.exists():
                logger.error(f"{name} file not found at {path}")
                raise RuntimeError(f"Missing artifact: {name}")
            logger.info(f"  ✓ {name}: {path}")
        logger.info(f"[2/5] Model paths OK ({time.time() - step_start:.2f}s)")
    except Exception:
        logger.error(f"[2/5] FAILED:\n{traceback.format_exc()}")
        raise

    # --- Step 3: Load ML Service (lazy init) ---
    try:
        step_start = time.time()
        logger.info("[3/5] Loading ML service (model + SHAP explainer)...")
        sys.stdout.flush()

        from api.services.ml_service import get_ml_service

        ml_service = get_ml_service()

        if not hasattr(ml_service.model, "predict_proba"):
            logger.error("Loaded model does not support predict_proba.")
            raise RuntimeError("Invalid model format.")

        if ml_service.explainer is None:
            logger.error("SHAP explainer failed to initialize.")
            raise RuntimeError("Invalid SHAP configuration.")

        logger.info(f"[3/5] ML service OK ({time.time() - step_start:.2f}s)")
    except Exception:
        logger.error(f"[3/5] FAILED:\n{traceback.format_exc()}")
        raise

    # --- Step 4: Setup Database Tables ---
    try:
        step_start = time.time()
        logger.info("[4/5] Creating database tables...")
        from api.database.session import Base, SessionLocal, engine

        Base.metadata.create_all(bind=engine)
        logger.info(f"[4/5] Database tables OK ({time.time() - step_start:.2f}s)")
    except Exception:
        logger.error(f"[4/5] FAILED:\n{traceback.format_exc()}")
        raise

    # --- Step 5: Create Default Admin User ---
    try:
        step_start = time.time()
        logger.info("[5/5] Ensuring default admin user exists...")
        from api.auth.auth_service import auth_service

        db = SessionLocal()
        try:
            auth_service.create_default_admin(db)
        finally:
            db.close()
        logger.info(f"[5/5] Admin user OK ({time.time() - step_start:.2f}s)")
    except Exception:
        logger.error(f"[5/5] FAILED:\n{traceback.format_exc()}")
        raise

    total = time.time() - startup_start
    logger.info("=" * 60)
    logger.info(f"STARTUP COMPLETE in {total:.2f}s — ready to serve requests")
    logger.info("=" * 60)
    sys.stdout.flush()

    yield

    logger.info("Shutting down API...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Enterprise API for Employee Attrition Prediction System",
    lifespan=lifespan,
    swagger_ui_init_oauth={
        "usePkceWithAuthorizationCodeGrant": False,
        "clientId": "swagger-ui",
    },
    swagger_ui_parameters={
        "persistAuthorization": True,
    },
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


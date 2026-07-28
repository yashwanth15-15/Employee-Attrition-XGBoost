from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import time

from api.config import settings
from api.core.error_handlers import add_exception_handlers
from api.core.logger import logger

from api.routes import predict, simulate, analytics, reports, health

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Enterprise API for Employee Attrition Prediction System"
)

# CORS Configuration
origins = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:8000",
    "http://127.0.0.1:8000"
]

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
    logger.info(f"Response status: {response.status_code} - Execution time: {process_time:.4f}s")
    return response

# Exception handlers
add_exception_handlers(app)

# Routes
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
        "health": f"{settings.API_V1_STR}/health"
    }

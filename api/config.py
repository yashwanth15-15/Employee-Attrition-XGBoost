import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Employee Attrition API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DB_PATH: str = os.path.join(BASE_DIR, "employee_predictions.db")
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")
    
    # Model files
    XGBOOST_MODEL_PATH: str = os.path.join(MODELS_DIR, "final_xgboost_model.pkl")
    ENCODERS_PATH: str = os.path.join(MODELS_DIR, "final_encoders.pkl")
    FEATURES_PATH: str = os.path.join(MODELS_DIR, "final_features.pkl")

settings = Settings()

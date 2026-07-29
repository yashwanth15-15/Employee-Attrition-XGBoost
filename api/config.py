import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """API Configuration settings."""

    PROJECT_NAME: str = "Employee Attrition API"
    VERSION: str = "4.0.0"
    API_V1_STR: str = "/api/v1"
    ALLOWED_ORIGINS: str = os.environ.get(
        "ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
    )

    # Paths (Resolved via pathlib)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATABASE_PATH: Path = Path(
        os.environ.get("DATABASE_PATH", BASE_DIR / "employee_predictions.db")
    )
    MODELS_DIR: Path = BASE_DIR / "models"

    # Model files
    XGBOOST_MODEL_PATH: Path = MODELS_DIR / "final_xgboost_model.pkl"
    ENCODERS_PATH: Path = MODELS_DIR / "final_encoders.pkl"
    FEATURES_PATH: Path = MODELS_DIR / "final_features.pkl"

    # Authentication Config
    SECRET_KEY: str = os.environ.get(
        "SECRET_KEY", "b336ff9c9e821b0333246ebde510ec0609591fc35728de7da62f7971df26f1c7"
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()

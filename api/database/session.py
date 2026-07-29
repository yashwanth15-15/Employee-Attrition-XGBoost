from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from api.config import settings
from api.core.logger import logger

DATABASE_URL = settings.DATABASE_URL
logger.info(f"Database URL resolved to: {DATABASE_URL}")

# Ensure the parent directory for SQLite exists (Render containers may lack it)
if DATABASE_URL.startswith("sqlite"):
    db_path = Path(str(settings.DATABASE_PATH))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured database directory exists: {db_path.parent}")

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

logger.info("Creating SQLAlchemy engine...")
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
logger.info("SQLAlchemy engine and SessionLocal created successfully.")

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


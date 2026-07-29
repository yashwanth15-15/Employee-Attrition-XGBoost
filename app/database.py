import os
import sys
import pandas as pd
from sqlalchemy import text

# Add the root directory to path to import api modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.database.session import engine, Base, SessionLocal
from api.database.models import PredictionHistory

def get_connection():
    # Deprecated for raw queries, but returning engine for backward compat
    return engine.connect()

def create_table():
    """Delegated to SQLAlchemy."""
    Base.metadata.create_all(bind=engine)

def ensure_schema():
    """Handled by SQLAlchemy."""
    pass

def add_prediction(record: dict):
    """Insert a prediction record into the database using SQLAlchemy."""
    with SessionLocal() as db:
        db_record = PredictionHistory(**record)
        db.add(db_record)
        db.commit()
        db.refresh(db_record)
        return db_record.id

def get_predictions(filters: dict = None) -> pd.DataFrame:
    """Retrieve predictions with optional filtering using pandas read_sql."""
    query = "SELECT * FROM predictions"
    params = {}
    conditions = []
    
    if filters:
        if filters.get("department"):
            conditions.append("department IN :department")
            params["department"] = tuple(filters["department"])
        if filters.get("risk_category"):
            conditions.append("risk_category IN :risk_category")
            params["risk_category"] = tuple(filters["risk_category"])
        if filters.get("start_date"):
            # PostgreSQL requires casting string to date for comparison, 
            # SQLite date() is different. For simplicity and cross-compatibility, 
            # we do basic string comparison since format is YYYY-MM-DD
            conditions.append("prediction_date >= :start_date")
            params["start_date"] = filters["start_date"]
        if filters.get("end_date"):
            conditions.append("prediction_date <= :end_date")
            params["end_date"] = filters["end_date"] + " 23:59:59"
        if filters.get("min_age") is not None:
            conditions.append("age >= :min_age")
            params["min_age"] = filters["min_age"]
        if filters.get("max_age") is not None:
            conditions.append("age <= :max_age")
            params["max_age"] = filters["max_age"]

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    # Use Pandas to read SQL securely with parameters
    df = pd.read_sql(text(query), con=engine.connect(), params=params)
    return df

def delete_predictions(ids: list):
    """Delete predictions by a list of ids."""
    if not ids:
        return
    with SessionLocal() as db:
        db.query(PredictionHistory).filter(PredictionHistory.id.in_(ids)).delete(synchronize_session=False)
        db.commit()

def delete_all_predictions():
    with SessionLocal() as db:
        db.query(PredictionHistory).delete(synchronize_session=False)
        db.commit()

def get_user_by_username(username: str):
    from api.database.crud import get_user_by_username as crud_get_user
    with SessionLocal() as db:
        user = crud_get_user(db, username)
        if user:
            return {c.name: getattr(user, c.name) for c in user.__table__.columns}
        return None

def create_user(user: dict):
    from api.database.crud import create_user as crud_create_user
    with SessionLocal() as db:
        crud_create_user(db, user)

def prediction_exists_at_datetime(dt_str: str, unique_fields: dict = None) -> bool:
    """Check if a prediction with the same datetime already exists."""
    query = "SELECT 1 FROM predictions WHERE prediction_date = :dt_str"
    params = {"dt_str": dt_str}
    
    if unique_fields:
        for key, value in unique_fields.items():
            query += f" AND {key} = :{key}"
            params[key] = value
            
    with engine.connect() as conn:
        result = conn.execute(text(query), params).fetchone()
        return result is not None

def seed_database_if_empty():
    import random
    from datetime import datetime, timedelta

    df = get_predictions()
    if not df.empty:
        return

    departments = ["Sales", "Human Resources", "Research & Development"]
    genders = ["Male", "Female"]
    marital_statuses = ["Single", "Married", "Divorced"]
    overtimes = ["Yes", "No"]

    for i in range(1, 21):
        prob = random.uniform(0.1, 0.9)
        risk = "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"
        health = int((1 - prob) * 100)
        income = random.randint(3000, 15000)

        record = {
            "prediction_date": (
                datetime.now() - timedelta(days=random.randint(0, 30))
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "employee_id": f"EMP{i:03d}",
            "employee_name": f"Employee {i}",
            "age": random.randint(22, 60),
            "gender": random.choice(genders),
            "department": random.choice(departments),
            "marital_status": random.choice(marital_statuses),
            "monthly_income": income,
            "years_at_company": random.randint(0, 20),
            "job_satisfaction": random.randint(1, 4),
            "work_life_balance": random.randint(1, 4),
            "overtime": random.choice(overtimes),
            "prediction_probability": prob,
            "risk_category": risk,
            "health_score": health,
            "replacement_cost": float(income * 12),
            "shap_summary": "Monthly Income: 15.0%, OverTime: 12.0%, Age: -5.0%",
        }
        add_prediction(record)

# Ensure table exists on import
create_table()
seed_database_if_empty()

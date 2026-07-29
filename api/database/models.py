from sqlalchemy import Boolean, Column, Float, Integer, String
from api.database.session import Base

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    role = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(String, nullable=False)


class PredictionHistory(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    prediction_date = Column(String, index=True)
    employee_id = Column(String, index=True)
    employee_name = Column(String)
    age = Column(Integer)
    gender = Column(String)
    department = Column(String)
    marital_status = Column(String)
    monthly_income = Column(Float)
    years_at_company = Column(Integer)
    job_satisfaction = Column(Integer)
    work_life_balance = Column(Integer)
    overtime = Column(String)
    prediction_probability = Column(Float)
    risk_category = Column(String)
    health_score = Column(Integer)
    replacement_cost = Column(Float)
    shap_summary = Column(String)


class SimulationHistory(Base):
    __tablename__ = "simulations"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    simulation_date = Column(String, index=True)
    employee_id = Column(String, index=True)
    original_probability = Column(Float)
    new_probability = Column(Float)
    probability_change = Column(Float)
    original_risk = Column(String)
    new_risk = Column(String)
    impact_analysis = Column(String)

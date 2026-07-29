from sqlalchemy.orm import Session
from api.database.models import User, PredictionHistory, SimulationHistory

def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def create_user(db: Session, user: dict):
    db_user = User(**user)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def create_prediction(db: Session, record: dict) -> int:
    db_record = PredictionHistory(**record)
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return db_record.id

def get_all_predictions(db: Session):
    return db.query(PredictionHistory).all()

def create_simulation(db: Session, record: dict) -> int:
    db_record = SimulationHistory(**record)
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return db_record.id

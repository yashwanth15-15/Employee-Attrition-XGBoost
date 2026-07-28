import sqlite3
import pandas as pd
from datetime import datetime

DATABASE = "employee_predictions.db"

EXPECTED_COLUMNS = {
    "prediction_date": "TEXT",
    "employee_name": "TEXT",
    "age": "INTEGER",
    "gender": "TEXT",
    "department": "TEXT",
    "marital_status": "TEXT",
    "monthly_income": "REAL",
    "years_at_company": "INTEGER",
    "job_satisfaction": "INTEGER",
    "work_life_balance": "INTEGER",
    "overtime": "TEXT",
    "prediction_probability": "REAL",
    "risk_category": "TEXT",
    "health_score": "INTEGER",
    "replacement_cost": "REAL",
}


def get_connection():
    return sqlite3.connect(DATABASE)


def create_table():
    """Create the predictions table with the latest schema if it does not exist."""
    conn = get_connection()
    cursor = conn.cursor()
    columns_def = ",\n            ".join([f"{col} {ctype}" for col, ctype in EXPECTED_COLUMNS.items()])
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {columns_def}
        )
        """
    )
    conn.commit()
    conn.close()


def ensure_schema():
    """Add missing columns to the predictions table safely.
    This function is idempotent and can be called on every import.
    It ensures all required columns exist (adds any that are missing) without
    dropping or recreating the table.
    """
    conn = get_connection()
    cursor = conn.cursor()
    # Retrieve existing column names
    cursor.execute("PRAGMA table_info(predictions)")
    existing = {row[1] for row in cursor.fetchall()}
    
    for col, col_type in EXPECTED_COLUMNS.items():
        if col not in existing:
            cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col} {col_type}")
    conn.commit()
    conn.close()


def add_prediction(record: dict):
    """Insert a prediction record into the database."""
    # Validate columns
    for key in record:
        if key not in EXPECTED_COLUMNS and key != 'id':
            raise ValueError(f"Unknown column '{key}' provided in prediction record.")

    conn = get_connection()
    cursor = conn.cursor()
    columns = ', '.join(record.keys())
    placeholders = ', '.join(['?'] * len(record))
    cursor.execute(f"INSERT INTO predictions ({columns}) VALUES ({placeholders})", tuple(record.values()))
    pred_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return pred_id


def get_predictions(filters: dict = None) -> pd.DataFrame:
    """Retrieve predictions with optional filtering.
    Filters keys: department (list), risk_category (list), start_date, end_date, min_age, max_age
    """
    conn = get_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM predictions"
    params = []
    conditions = []
    if filters:
        if filters.get('department'):
            placeholders = ','.join(['?'] * len(filters['department']))
            conditions.append(f"department IN ({placeholders})")
            params.extend(filters['department'])
        if filters.get('risk_category'):
            placeholders = ','.join(['?'] * len(filters['risk_category']))
            conditions.append(f"risk_category IN ({placeholders})")
            params.extend(filters['risk_category'])
        if filters.get('start_date'):
            conditions.append("date(prediction_date) >= date(?)")
            params.append(filters['start_date'])
        if filters.get('end_date'):
            conditions.append("date(prediction_date) <= date(?)")
            params.append(filters['end_date'])
        if filters.get('min_age') is not None:
            conditions.append("age >= ?")
            params.append(filters['min_age'])
        if filters.get('max_age') is not None:
            conditions.append("age <= ?")
            params.append(filters['max_age'])
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()
    col_names = [description[0] for description in cursor.description]
    conn.close()
    return pd.DataFrame(rows, columns=col_names)


def delete_predictions(ids: list):
    """Delete predictions by a list of ids."""
    if not ids:
        return
    conn = get_connection()
    cursor = conn.cursor()
    placeholders = ','.join(['?'] * len(ids))
    cursor.execute(f"DELETE FROM predictions WHERE id IN ({placeholders})", tuple(ids))
    conn.commit()
    conn.close()


def delete_all_predictions():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()


def prediction_exists_at_datetime(dt_str: str, unique_fields: dict = None) -> bool:
    """Check if a prediction with the same datetime already exists.
    Optionally match additional fields to avoid duplicates.
    """
    conn = get_connection()
    cursor = conn.cursor()
    query = "SELECT 1 FROM predictions WHERE prediction_date = ?"
    params = [dt_str]
    if unique_fields:
        for key, value in unique_fields.items():
            query += f" AND {key} = ?"
            params.append(value)
    cursor.execute(query, tuple(params))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


# Ensure table exists on import
# Ensure schema is up‑to‑date on import
create_table()
ensure_schema()
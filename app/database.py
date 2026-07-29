import sqlite3

import pandas as pd

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
    "employee_id": "TEXT",
    "shap_summary": "TEXT",
}


def get_connection():
    return sqlite3.connect(DATABASE)


def create_table():
    """Create the predictions table with the latest schema if it does not exist."""
    conn = get_connection()
    cursor = conn.cursor()
    columns_def = ",\n            ".join(
        [f"{col} {ctype}" for col, ctype in EXPECTED_COLUMNS.items()]
    )
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {columns_def}
        )
        """)
    
    # Create Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE,
            hashed_password TEXT NOT NULL,
            role TEXT NOT NULL,
            is_active BOOLEAN NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        )
    """)
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
        if key not in EXPECTED_COLUMNS and key != "id":
            raise ValueError(f"Unknown column '{key}' provided in prediction record.")

    conn = get_connection()
    cursor = conn.cursor()
    columns = ", ".join(record.keys())
    placeholders = ", ".join(["?"] * len(record))
    cursor.execute(
        f"INSERT INTO predictions ({columns}) VALUES ({placeholders})",
        tuple(record.values()),
    )
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
        if filters.get("department"):
            placeholders = ",".join(["?"] * len(filters["department"]))
            conditions.append(f"department IN ({placeholders})")
            params.extend(filters["department"])
        if filters.get("risk_category"):
            placeholders = ",".join(["?"] * len(filters["risk_category"]))
            conditions.append(f"risk_category IN ({placeholders})")
            params.extend(filters["risk_category"])
        if filters.get("start_date"):
            conditions.append("date(prediction_date) >= date(?)")
            params.append(filters["start_date"])
        if filters.get("end_date"):
            conditions.append("date(prediction_date) <= date(?)")
            params.append(filters["end_date"])
        if filters.get("min_age") is not None:
            conditions.append("age >= ?")
            params.append(filters["min_age"])
        if filters.get("max_age") is not None:
            conditions.append("age <= ?")
            params.append(filters["max_age"])
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
    placeholders = ",".join(["?"] * len(ids))
    cursor.execute(f"DELETE FROM predictions WHERE id IN ({placeholders})", tuple(ids))
    conn.commit()
    conn.close()


def delete_all_predictions():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()


def get_user_by_username(username: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "id": row[0],
            "username": row[1],
            "email": row[2],
            "hashed_password": row[3],
            "role": row[4],
            "is_active": bool(row[5]),
            "created_at": row[6]
        }
    return None

def create_user(user: dict):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO users (id, username, email, hashed_password, role, is_active, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (user['id'], user['username'], user.get('email'), user['hashed_password'], user['role'], user.get('is_active', True), user['created_at'])
    )
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
            "replacement_cost": income * 12,
            "shap_summary": "Monthly Income: 15.0%, OverTime: 12.0%, Age: -5.0%",
        }
        add_prediction(record)


# Ensure table exists on import
# Ensure schema is up‑to‑date on import
create_table()
ensure_schema()
seed_database_if_empty()

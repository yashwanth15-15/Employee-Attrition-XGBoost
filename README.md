# Employee Attrition Prediction & HR Analytics Platform

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-00a393.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-blue.svg)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0+-red.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-336791.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Project Overview
An enterprise AI-powered HR analytics platform using Machine Learning, Explainable AI, FastAPI, and Streamlit. This application predicts employee attrition risk and provides actionable HR recommendations.

## Key Features
- Employee Attrition Prediction
- Batch Prediction
- What-If Simulation
- SHAP Explainable AI
- HR Recommendation Engine
- Prediction History
- Executive Dashboard
- Department Analytics
- Employee Comparison
- Workforce Planning
- PDF Report Generation
- REST API
- JWT Authentication
- Role-Based Access Control (RBAC)
- SQLAlchemy ORM
- SQLite (Local)
- PostgreSQL (Production)
- Docker Support

## Technology Stack

**Frontend:**
- Streamlit

**Backend:**
- FastAPI

**Machine Learning:**
- XGBoost
- SHAP

**Database:**
- SQLAlchemy ORM
- SQLite
- PostgreSQL

**Authentication:**
- JWT
- bcrypt
- OAuth2

**Visualization:**
- Plotly

**Deployment:**
- Docker
- Render

**Language:**
- Python

## Architecture

User
↓
Streamlit Dashboard
↓
FastAPI REST API
↓
JWT Authentication
↓
Role-Based Access Control
↓
SQLAlchemy ORM
↓
SQLite (Local) / PostgreSQL (Production)
↓
XGBoost Model
↓
SHAP Explainability

## Folder Structure

```
├── api/                  # FastAPI backend
│   ├── auth/             # Authentication mechanisms
│   ├── core/             # Application config and logger
│   ├── database/         # SQLAlchemy ORM schemas and connection
│   ├── models/           # Pydantic validation schemas
│   ├── routes/           # REST endpoints
│   └── services/         # Machine Learning and DB services
├── App/                  # Streamlit frontend
│   ├── views/            # Dashboard pages
│   ├── app.py            # Streamlit entry point
│   └── auth_ui.py        # Streamlit auth interface
├── docs/                 # Documentation
├── data/                 # Datasets
├── tests/                # Pytest suites
└── requirements.txt      # Dependencies
```

## Installation

```bash
# Clone repository
git clone https://github.com/yashwanth15-15/Employee-Attrition-XGBoost.git
cd Employee-Attrition-XGBoost

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install requirements
pip install -r requirements.txt

# Start FastAPI (Terminal 1)
uvicorn api.main:app --reload

# Start Streamlit (Terminal 2)
streamlit run App/app.py
```

## API Endpoints
- **Authentication**: `/api/v1/auth/login`, `/api/v1/auth/me`
- **Health**: `/api/v1/health`
- **Prediction**: `/api/v1/predict`
- **Simulation**: `/api/v1/simulate`
- **Analytics**: `/api/v1/analytics/summary`
- **Reports**: `/api/v1/reports/pdf`

## Screenshots Section
*(Placeholders for future screenshots)*
- Login Page
- Dashboard
- Prediction
- SHAP
- Analytics
- Executive Dashboard

## Future Enhancements
- AI HR Copilot
- Email Notifications
- Password Recovery
- Multi-user Management
- LLM-powered Insights
- Cloud Deployment

## License
MIT License

## Author
Bankapalli Yashwanth

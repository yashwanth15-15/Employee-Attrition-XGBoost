# API Documentation v2.0.0

## Authentication Endpoints
- `POST /api/v1/auth/login`: Issue JWT token
- `GET /api/v1/auth/me`: Retrieve current user profile

## Core Endpoints
- `GET /api/v1/health`: Check database and model status
- `POST /api/v1/predict`: Submit features for XGBoost prediction
- `POST /api/v1/simulate`: Run what-if simulations
- `GET /api/v1/analytics/summary`: Fetch aggregated HR analytics
- `GET /api/v1/reports/pdf`: Generate PDF report payload

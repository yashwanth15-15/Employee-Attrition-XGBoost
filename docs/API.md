# API Documentation

The backend exposes a RESTful API powered by FastAPI, fully documented via OpenAPI/Swagger.

## Accessing Interactive Docs
When the backend is running locally, access the interactive Swagger UI at:
👉 `http://localhost:8000/docs`

Or the ReDoc alternative at:
👉 `http://localhost:8000/redoc`

---

## Key Endpoints

### 1. Predict Attrition Risk
**`POST /api/v1/predict`**
Evaluates an employee's profile to determine their probability of leaving the company, extracting the driving factors via SHAP, and querying the HR rule engine for recommended actions.

**Example Payload:**
```json
{
  "Age": 30,
  "Gender": "Male",
  "Department": "Sales",
  "Monthly Income": 5000,
  "Marital Status": "Single",
  "OverTime": "No",
  "Years At Company": 5,
  "Total Working Years": 8,
  "Job Satisfaction": 3,
  "Environment Satisfaction": 3,
  "Work Life Balance": 3,
  "Years Since Last Promotion": 0,
  "Training Times Last Year": 2,
  "Business Travel": "Travel_Rarely",
  "Distance From Home": 5,
  "Performance Rating": 3,
  "Stock Option Level": 0,
  "Years In Current Role": 3
}
```

**Response Data:**
- `probability`: Float (0.0 to 1.0)
- `risk_category`: String ("Low", "Medium", "High")
- `shap_values`: Key-Value mapping of features to their % contribution.
- `recommendations`: Actionable steps divided into Immediate, Medium Term, and Long Term strategies.
- `prediction_id`: The ID of the record successfully saved to the SQLite database.

---

### 2. What-If Simulator
**`POST /api/v1/simulate`**
Temporarily alters an employee's profile to observe how specific interventions (like increasing salary or reducing overtime) impact their attrition probability.

**Example Payload:**
```json
{
  "base_features": { /* Full employee object */ },
  "modified_features": {
      "OverTime": "No",
      "Monthly Income": 6500
  }
}
```
**Response Data:**
- `probability_change`: The delta between the original and new prediction.
- `impact_analysis`: Textual summary of the intervention's success.

---

### 3. Department Analytics
**`GET /api/v1/analytics/summary`**
Aggregates all historical predictions from the SQLite database to generate high-level KPI metrics used by the Executive Dashboard.

---

### 4. Health Check
**`GET /api/v1/health`**
Returns the status of the API, verifying that the XGBoost model is loaded into memory and the database connection is active.

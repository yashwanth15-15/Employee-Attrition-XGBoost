from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class EmployeeFeatures(BaseModel):
    Age: int = Field(..., ge=18, le=100)
    Gender: Literal["Male", "Female"] = Field(..., description="Male or Female")
    Department: Literal["Sales", "Research & Development", "Human Resources"] = Field(
        ..., description="Department name"
    )
    Monthly_Income: float = Field(..., alias="Monthly Income")
    Marital_Status: Literal["Single", "Married", "Divorced"] = Field(
        ..., alias="Marital Status"
    )
    OverTime: Literal["Yes", "No"] = Field(..., description="Yes or No")
    Years_At_Company: int = Field(..., alias="Years At Company", ge=0)
    Total_Working_Years: int = Field(..., alias="Total Working Years", ge=0)
    Job_Satisfaction: int = Field(..., alias="Job Satisfaction", ge=1, le=4)
    Environment_Satisfaction: int = Field(
        ..., alias="Environment Satisfaction", ge=1, le=4
    )
    Work_Life_Balance: int = Field(..., alias="Work Life Balance", ge=1, le=4)
    Years_Since_Last_Promotion: int = Field(
        ..., alias="Years Since Last Promotion", ge=0
    )
    Training_Times_Last_Year: int = Field(..., alias="Training Times Last Year", ge=0)
    Business_Travel: Literal["Non-Travel", "Travel_Rarely", "Travel_Frequently"] = (
        Field(..., alias="Business Travel")
    )
    Distance_From_Home: int = Field(..., alias="Distance From Home", ge=0)
    Performance_Rating: int = Field(3, alias="Performance Rating", ge=1, le=4)
    Stock_Option_Level: int = Field(0, alias="Stock Option Level", ge=0, le=3)
    Years_In_Current_Role: int = Field(0, alias="Years In Current Role", ge=0)

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "Age": 30,
                "Gender": "Male",
                "Department": "Sales",
                "Monthly Income": 5000.0,
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
                "Years In Current Role": 3,
            }
        },
    }


class HRRecommendations(BaseModel):
    priority: str
    response_time: str
    factors: List[str]
    immediate_actions: List[str]
    medium_term_actions: List[str]
    long_term_strategy: List[str]
    strategies_text: List[str]


class RiskDriver(BaseModel):
    feature: str
    contribution: float
    impact: str
    shap_value: float


class PredictionResponse(BaseModel):
    probability: float
    risk_category: str
    top_risk_drivers: List[RiskDriver]
    recommendations: HRRecommendations
    prediction_id: Optional[int] = None


class SimulationRequest(BaseModel):
    base_features: EmployeeFeatures
    modified_features: Dict[str, Any]


class SimulationResponse(BaseModel):
    original_probability: float
    new_probability: float
    probability_change: float
    original_risk: str
    new_risk: str
    impact_analysis: str

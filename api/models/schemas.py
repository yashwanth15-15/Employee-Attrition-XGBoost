from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class EmployeeFeatures(BaseModel):
    """Schema representing employee data features required for prediction."""
    
    Age: int = Field(..., ge=18, le=100, description="Employee's age in years.")
    Gender: Literal["Male", "Female"] = Field(..., description="Male or Female")
    Department: Literal["Sales", "Research & Development", "Human Resources"] = Field(
        ..., description="Department name"
    )
    Monthly_Income: float = Field(..., alias="Monthly Income", description="Monthly salary in USD.")
    Marital_Status: Literal["Single", "Married", "Divorced"] = Field(
        ..., alias="Marital Status"
    )
    OverTime: Literal["Yes", "No"] = Field(..., description="Yes or No")
    Years_At_Company: int = Field(..., alias="Years At Company", ge=0, description="Number of years employed at the company.")
    Total_Working_Years: int = Field(..., alias="Total Working Years", ge=0, description="Total years of professional experience.")
    Job_Satisfaction: int = Field(..., alias="Job Satisfaction", ge=1, le=4, description="Job satisfaction rating (1-4).")
    Environment_Satisfaction: int = Field(
        ..., alias="Environment Satisfaction", ge=1, le=4, description="Work environment satisfaction rating (1-4)."
    )
    Work_Life_Balance: int = Field(..., alias="Work Life Balance", ge=1, le=4, description="Work-life balance rating (1-4).")
    Years_Since_Last_Promotion: int = Field(
        ..., alias="Years Since Last Promotion", ge=0, description="Years since the last promotion."
    )
    Training_Times_Last_Year: int = Field(..., alias="Training Times Last Year", ge=0, description="Number of training sessions attended last year.")
    Business_Travel: Literal["Non-Travel", "Travel_Rarely", "Travel_Frequently"] = (
        Field(..., alias="Business Travel", description="Frequency of business travel.")
    )
    Distance_From_Home: int = Field(..., alias="Distance From Home", ge=0, description="Distance from home in miles/km.")
    Performance_Rating: int = Field(3, alias="Performance Rating", ge=1, le=4, description="Employee performance rating (1-4).")
    Stock_Option_Level: int = Field(0, alias="Stock Option Level", ge=0, le=3, description="Stock options level granted (0-3).")
    Years_In_Current_Role: int = Field(0, alias="Years In Current Role", ge=0, description="Years spent in the current role.")

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
    """Schema for HR actionable recommendations based on risk."""
    priority: str = Field(..., description="Action priority level (e.g., High, Medium, Low).")
    response_time: str
    factors: List[str]
    immediate_actions: List[str]
    medium_term_actions: List[str]
    long_term_strategy: List[str]
    strategies_text: List[str]


class RiskDriver(BaseModel):
    """Schema representing a single feature's contribution to attrition risk."""
    feature: str = Field(..., description="Name of the feature.")
    contribution: float
    impact: str
    shap_value: float


class PredictionResponse(BaseModel):
    """Schema for the main prediction endpoint response."""
    probability: float = Field(..., description="Calculated probability of attrition (0.0 to 1.0).")
    risk_category: str = Field(..., description="Assigned risk category based on probability.")
    top_risk_drivers: List[RiskDriver]
    recommendations: HRRecommendations
    prediction_id: Optional[int] = None


class SimulationRequest(BaseModel):
    """Schema for requesting a what-if analysis on modified employee features."""
    base_features: EmployeeFeatures = Field(..., description="Original employee features.")
    modified_features: Dict[str, Any]


class SimulationResponse(BaseModel):
    """Schema for the response to a what-if simulation request."""
    original_probability: float = Field(..., description="Probability of attrition before modifications.")
    new_probability: float
    probability_change: float
    original_risk: str
    new_risk: str
    impact_analysis: str

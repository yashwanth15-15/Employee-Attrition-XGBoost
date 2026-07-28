from fastapi import APIRouter
from api.models.schemas import SimulationRequest, SimulationResponse, EmployeeFeatures
from api.services.ml_service import ml_service

router = APIRouter(prefix="/simulate", tags=["Simulation"])

@router.post("", response_model=SimulationResponse, summary="Simulate What-If Scenarios", description="Modify features of an employee to see how it affects their attrition probability.")
async def simulate_what_if(request: SimulationRequest):
    base_dict = request.base_features.model_dump(by_alias=True)
    
    # Predict original
    orig_prob, orig_risk, _ = ml_service.predict(base_dict)
    
    # Apply modifications securely by validating against the schema
    base_dict.update(request.modified_features)
    mod_employee = EmployeeFeatures(**base_dict)
    mod_dict = mod_employee.model_dump(by_alias=True)
            
    # Predict new
    new_prob, new_risk, _ = ml_service.predict(mod_dict)
    
    delta = orig_prob - new_prob
    if delta > 0:
        impact = f"Intervention improves retention probability by {delta*100:.1f}%"
    elif delta < 0:
        impact = f"Intervention increases attrition probability by {abs(delta)*100:.1f}%"
    else:
        impact = "No change in attrition probability."
        
    return SimulationResponse(
        original_probability=orig_prob,
        new_probability=new_prob,
        probability_change=delta,
        original_risk=orig_risk,
        new_risk=new_risk,
        impact_analysis=impact
    )

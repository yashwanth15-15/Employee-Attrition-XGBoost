from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.auth.dependencies import RoleChecker
from api.database.session import get_db

from api.core.exceptions import APIException
from api.core.logger import logger
from api.models.schemas import (EmployeeFeatures, SimulationRequest,
                                SimulationResponse)
from api.services.ml_service import ml_service

router = APIRouter(
    prefix="/simulate", 
    tags=["Simulation"],
    dependencies=[Depends(RoleChecker(["Admin", "HR_Manager"]))]
)


@router.post(
    "",
    response_model=SimulationResponse,
    summary="Simulate What-If Scenarios",
    description="Modify features of an employee to see how it affects their attrition probability.",
)
async def simulate_what_if(
    request: SimulationRequest,
    db: Session = Depends(get_db)
) -> SimulationResponse:
    """
    Simulates changes to an employee's profile and returns the delta in attrition probability.
    """
    logger.info("Request received for /simulate")
    try:
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
            impact = (
                f"Intervention increases attrition probability by {abs(delta)*100:.1f}%"
            )
        else:
            impact = "No change in attrition probability."

        logger.info("Successfully processed /simulate")
        return SimulationResponse(
            original_probability=orig_prob,
            new_probability=new_prob,
            probability_change=delta,
            original_risk=orig_risk,
            new_risk=new_risk,
            impact_analysis=impact,
        )
    except APIException as e:
        logger.error(f"APIException in /simulate: {e.message}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /simulate: {str(e)}")
        raise APIException(f"Simulation failed: {str(e)}", status_code=500)

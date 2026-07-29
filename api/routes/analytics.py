from typing import Any, Dict

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.auth.dependencies import RoleChecker
from api.database.session import get_db

from api.core.exceptions import APIException
from api.core.logger import logger
from api.services.db_service import db_service

router = APIRouter(
    prefix="/analytics", 
    tags=["Analytics"],
    dependencies=[Depends(RoleChecker(["Admin", "HR_Manager", "Viewer"]))]
)


@router.get(
    "/summary",
    response_model=Dict[str, Any],
    summary="Get Department Analytics",
    description="Retrieves aggregated risk statistics across all departments from the database.",
)
async def get_analytics_summary(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Retrieves and calculates high-level analytics for all departments.
    """
    logger.info("Request received for /analytics/summary")
    try:
        records = db_service.get_all_predictions(db)

        total = len(records)
        if total == 0:
            return {"total_predictions": 0, "high_risk_count": 0, "departments": {}}

        high_risk = sum(1 for r in records if r.get("risk_category") == "High")

        # Department breakdown
        dept_stats: Dict[str, Dict[str, int]] = {}
        for r in records:
            dept = r.get("department", "Unknown")
            if dept not in dept_stats:
                dept_stats[dept] = {"count": 0, "high_risk": 0}
            dept_stats[dept]["count"] += 1
            if r.get("risk_category") == "High":
                dept_stats[dept]["high_risk"] += 1

        logger.info("Successfully processed /analytics/summary")
        return {
            "total_predictions": total,
            "high_risk_count": high_risk,
            "high_risk_percentage": (high_risk / total) * 100,
            "departments": dept_stats,
        }
    except APIException as e:
        logger.error(f"APIException in /analytics/summary: {e.message}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /analytics/summary: {str(e)}")
        raise APIException(f"Failed to retrieve analytics: {str(e)}", status_code=500)

from typing import Any, Dict

from fastapi import APIRouter

from api.services.db_service import db_service

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get(
    "/summary",
    response_model=Dict[str, Any],
    summary="Get Department Analytics",
    description="Retrieves aggregated risk statistics across all departments from the database.",
)
async def get_analytics_summary():
    records = db_service.get_all_predictions()

    total = len(records)
    if total == 0:
        return {"total_predictions": 0, "high_risk_count": 0, "departments": {}}

    high_risk = sum(1 for r in records if r.get("risk_category") == "High")

    # Department breakdown
    dept_stats = {}
    for r in records:
        dept = r.get("department", "Unknown")
        if dept not in dept_stats:
            dept_stats[dept] = {"count": 0, "high_risk": 0}
        dept_stats[dept]["count"] += 1
        if r.get("risk_category") == "High":
            dept_stats[dept]["high_risk"] += 1

    return {
        "total_predictions": total,
        "high_risk_count": high_risk,
        "high_risk_percentage": (high_risk / total) * 100,
        "departments": dept_stats,
    }

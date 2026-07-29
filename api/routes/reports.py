from io import BytesIO

import pandas as pd
from fastapi import APIRouter
from fastapi.responses import Response
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from api.core.exceptions import APIException
from api.core.logger import logger
from api.services.db_service import db_service

router = APIRouter(prefix="/reports", tags=["Reporting"])


@router.get(
    "/pdf",
    response_class=Response,
    summary="Download Analytics PDF",
    description="Generates a downloadable PDF report summarizing current employee attrition risks.",
)
async def generate_pdf_report() -> Response:
    """
    Generates a PDF report containing high-level analytics from the database.
    """
    logger.info("Request received for /reports/pdf")
    try:
        records = db_service.get_all_predictions()
        if not records:
            return Response(
                content="No data available", media_type="text/plain", status_code=404
            )

        df = pd.DataFrame(records)
        total = len(df)
        high_risk = len(df[df["risk_category"] == "High"])

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer)
        elements = []
        styles = getSampleStyleSheet()

        elements.append(Paragraph("Employee Attrition API Report", styles["Title"]))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph(f"Total Predictions: {total}", styles["Normal"]))
        elements.append(Paragraph(f"High Risk Employees: {high_risk}", styles["Normal"]))
        elements.append(Spacer(1, 12))

        doc.build(elements)
        buffer.seek(0)

        logger.info("Successfully processed /reports/pdf")
        return Response(
            content=buffer.read(),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=report.pdf"},
        )
    except APIException as e:
        logger.error(f"APIException in /reports/pdf: {e.message}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /reports/pdf: {str(e)}")
        raise APIException(f"Failed to generate PDF report: {str(e)}", status_code=500)

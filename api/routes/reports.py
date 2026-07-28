from fastapi import APIRouter
from fastapi.responses import Response
from api.services.db_service import db_service
import pandas as pd
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

router = APIRouter(prefix="/reports", tags=["Reporting"])

@router.get("/pdf", response_class=Response, summary="Download Analytics PDF", description="Generates a downloadable PDF report summarizing current employee attrition risks.")
async def generate_pdf_report():
    records = db_service.get_all_predictions()
    if not records:
        return Response(content="No data available", media_type="text/plain", status_code=404)
        
    df = pd.DataFrame(records)
    total = len(df)
    high_risk = len(df[df['risk_category'] == 'High'])
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []
    styles = getSampleStyleSheet()
    
    elements.append(Paragraph("Employee Attrition API Report", styles['Title']))
    elements.append(Spacer(1, 12))
    
    elements.append(Paragraph(f"Total Predictions: {total}", styles['Normal']))
    elements.append(Paragraph(f"High Risk Employees: {high_risk}", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    doc.build(elements)
    buffer.seek(0)
    
    return Response(content=buffer.read(), media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=report.pdf"})

def calculate_risk(probability: float) -> str:
    """
    Centralized risk calculation function.
    Returns the exact risk category (High, Medium, Low) based on the attrition probability.
    
    Thresholds:
    - High: >= 70%
    - Medium: >= 40%
    - Low: < 40%
    """
    if probability >= 0.70:
        return "High"
    elif probability >= 0.40:
        return "Medium"
    else:
        return "Low"

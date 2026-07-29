def generate_hr_recommendation(employee, probability, risk_category):
    """
    Generate professional HR recommendations based on employee features and risk metrics.
    Operates entirely offline using predefined HR intelligence rules.
    """

    actions = []
    strategies = []
    factors = []

    # Safe get for employee details to avoid KeyErrors
    def get_val(key, default=None):
        return employee.get(key, default)

    age = get_val("Age", 30)
    overtime = get_val("OverTime", "No")
    job_sat = get_val("Job Satisfaction", 3)
    env_sat = get_val("Environment Satisfaction", 3)
    wl_balance = get_val("Work Life Balance", 3)
    rel_sat = get_val("Relationship Satisfaction", 3)
    job_inv = get_val("Job Involvement", 3)
    yslp = get_val("Years Since Last Promotion", 0)
    income = get_val("Monthly Income", 5000)
    training = get_val("Training Times Last Year", 2)
    travel = get_val("Business Travel", "Non-Travel")
    distance = get_val("Distance From Home", 5)
    perf_rating = get_val("Performance Rating", 3)
    stock_options = get_val("Stock Option Level", 1)
    years_company = get_val("Years At Company", 2)
    years_curr_role = get_val("Years In Current Role", 1)

    # ---------------------------------------------------------
    # RULE SET 1: BASE RISK CATEGORY RULES (15 rules)
    # ---------------------------------------------------------
    if risk_category == "High":
        actions.extend(
            [
                "Immediate HR Review",
                "Manager Discussion",
                "Salary Review",
                "Career Development Plan",
                "Assign Mentor",
                "Retention Bonus Consideration",
                "Reduce Overtime",
            ]
        )
        strategies.append(
            "High priority intervention required. Schedule immediate meetings with direct manager and HR to address concerns."
        )
        if age < 30:
            actions.append("Career Mentoring Program")
            strategies.append(
                "Young employee at high risk: Implement fast-track career mentoring."
            )
        if age > 45:
            actions.append("Retention Planning")
            strategies.append(
                "Senior employee at high risk: Focus on long-term retention planning and knowledge transfer."
            )
    elif risk_category == "Medium":
        actions.extend(
            [
                "Monthly Check-ins",
                "Upskilling Program",
                "Employee Recognition",
                "Flexible Work Arrangement",
                "Wellness Program Enrollment",
            ]
        )
        strategies.append(
            "Monitor engagement levels and implement targeted retention activities to prevent escalation to high risk."
        )
    else:
        actions.extend(
            ["Continue Engagement", "Reward Performance", "Quarterly Monitoring"]
        )
        strategies.append(
            "Maintain current management practices while ensuring continuous career development opportunities."
        )

    # ---------------------------------------------------------
    # RULE SET 2: JOB SATISFACTION & ENVIRONMENT (8 rules)
    # ---------------------------------------------------------
    if job_sat <= 2:
        factors.append("Low Job Satisfaction")
        actions.append("Conduct engagement meeting")
        strategies.append(
            "Low job satisfaction detected. A 1-on-1 meeting is needed to understand specific pain points in their daily role."
        )
    if env_sat <= 2:
        factors.append("Poor Work Environment")
        actions.append("Improve workplace environment")
        strategies.append(
            "Employee is dissatisfied with work environment. Assess team dynamics and physical/virtual workspace conditions."
        )
    if rel_sat <= 2:
        factors.append("Low Relationship Satisfaction")
        actions.append("Team Building Activities")
        strategies.append(
            "Employee reports poor relationship satisfaction. Consider team building or manager mediation."
        )
    if job_inv <= 2:
        factors.append("Low Job Involvement")
        actions.append("Reassign or Rotate Role")
        strategies.append(
            "Employee feels disconnected from their work. Provide challenging and meaningful tasks."
        )

    # ---------------------------------------------------------
    # RULE SET 3: WORKLOAD & LIFESTYLE (8 rules)
    # ---------------------------------------------------------
    if overtime == "Yes":
        factors.append("Frequent Overtime")
        actions.append("Reduce overtime workload")
        strategies.append(
            "Employee is working overtime. Consider redistributing workload to prevent burnout."
        )
    if wl_balance <= 2:
        factors.append("Poor Work-Life Balance")
        actions.append("Offer flexible schedule")
        strategies.append(
            "Work-life balance is a concern. Introduce flexible working hours or remote days if applicable."
        )
    if travel == "Travel_Frequently":
        factors.append("High Travel Burden")
        actions.append("Review travel workload")
        strategies.append(
            "Frequent business travel may lead to fatigue. Consider assigning local projects or reducing travel requirements."
        )
    if distance > 15:
        factors.append("Long Commute")
        actions.append("Consider hybrid work")
        strategies.append(
            "Employee has a long commute. A hybrid or remote work arrangement could significantly improve retention."
        )

    # ---------------------------------------------------------
    # RULE SET 4: CAREER PROGRESSION & TRAINING (8 rules)
    # ---------------------------------------------------------
    if yslp > 5:
        factors.append("Stagnant Career Progression")
        actions.append("Initiate promotion review")
        strategies.append(
            "Employee has not been promoted in over 5 years. Evaluate for career advancement or lateral growth to prevent stagnation."
        )
    if training == 0:
        factors.append("Lack of Training")
        actions.append("Recommend mandatory training")
        strategies.append(
            "Lack of recent training. Enroll employee in skill development programs to boost engagement and capability."
        )
    if years_company > 5 and years_curr_role > 4 and yslp > 3:
        factors.append("Role Stagnation")
        actions.append("Role Rotation")
        strategies.append(
            "Employee has been in the same role for a long time. Consider lateral moves to provide new challenges."
        )

    # ---------------------------------------------------------
    # RULE SET 5: COMPENSATION & BENEFITS (6 rules)
    # ---------------------------------------------------------
    if income < 4000:
        factors.append("Low Monthly Income")
        actions.append("Compensation review")
        strategies.append(
            "Monthly income is relatively low. Benchmark salary against industry standards and consider a market adjustment."
        )
    if perf_rating >= 4 and income < 6000:
        factors.append("High Performer / Low Pay")
        actions.append("Salary Revision")
        strategies.append(
            "High performing employee with below-average compensation. Immediate salary review recommended to prevent poaching."
        )
    if stock_options == 0:
        factors.append("No Stock Options")
        actions.append("Review benefits package")
        strategies.append(
            "Employee has no stock options. Introducing long-term incentives can improve loyalty."
        )

    # Structuring the Output for Enterprise UI
    priority_level = (
        "High"
        if risk_category == "High"
        else "Medium" if risk_category == "Medium" else "Low"
    )

    if priority_level == "High":
        response_time = "Manager meeting within 7 days."
        if probability > 0.85:
            response_time = "Immediate HR intervention."
    elif priority_level == "Medium":
        response_time = "Review within 30 days."
    else:
        response_time = "Monitor every quarter."

    unique_actions = list(dict.fromkeys(actions))

    immediate_actions = []
    medium_term_actions = []
    long_term_strategy = []

    for action in unique_actions:
        if (
            "Immediate" in action
            or "Discussion" in action
            or "Review" in action
            or "Reduce" in action
            or "meeting" in action.lower()
        ):
            immediate_actions.append(action)
        elif (
            "Bonus" in action
            or "Program" in action
            or "Flexible" in action
            or "Hybrid" in action
            or "Training" in action
            or "Team Building" in action
        ):
            medium_term_actions.append(action)
        else:
            long_term_strategy.append(action)

    # Fallback categorization if empty
    if not immediate_actions and unique_actions:
        immediate_actions = unique_actions[:2]
        long_term_strategy = unique_actions[2:]

    # Fallbacks for empty lists
    if not medium_term_actions:
        medium_term_actions = ["Schedule follow-up feedback session"]
    if not long_term_strategy:
        long_term_strategy = ["Incorporate into annual retention planning"]

    return {
        "priority": priority_level,
        "response_time": response_time,
        "factors": factors,
        "immediate_actions": immediate_actions,
        "medium_term_actions": medium_term_actions,
        "long_term_strategy": long_term_strategy,
        "strategies_text": strategies[:4],
    }

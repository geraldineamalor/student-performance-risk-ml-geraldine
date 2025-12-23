def assign_risk(score):
    if score>=75:
        return "Low"
    elif score>=50:
        return "Medium"
    else:
        return "High"
import pandas as pd
from utils import assign_risk
data = pd.read_csv("data/student_performance.csv")
data["RiskLevel"] = data["FinalScore"].apply(assign_risk)
data.to_csv("data/student_performance_with_risk.csv", index=False)
print("RiskLevel column added and saved successfully")

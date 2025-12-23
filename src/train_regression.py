import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
import pickle
data = pd.read_csv("data/student_performance.csv")
X = data[["Attendance", "InternalMarks", "Assignments", "StudyHours"]]
y = data["FinalScore"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = math.sqrt(mse)
print("Regression Model Performance")
print("RMSE:", round(rmse, 2))
with open("model/score_model.pkl","wb") as file:
    pickle.dump(model,file)
print("Regression model saved as score_model.pkl")
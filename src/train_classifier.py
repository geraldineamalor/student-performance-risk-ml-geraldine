import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("data/student_performance_with_risk.csv")
X = data[["Attendance", "InternalMarks", "Assignments", "StudyHours"]]
y = data["RiskLevel"]
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)
print("Classification Model Performance")
print("Accuracy:", round(accuracy, 2))
print("Confusion Matrix:")
print(cm)
plt.figure(figsize=(6,4))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=encoder.classes_,yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix- Student Risk Classsification")
plt.savefig("assets/confusion_matrix.png",bbox_inches="tight")
plt.show()
with open("model/risk_model.pkl", "wb") as file:
    pickle.dump(model, file)
with open("model/risk_label_encoder.pkl", "wb") as file:
    pickle.dump(encoder, file)
print("Classification model and label encoder saved successfully")

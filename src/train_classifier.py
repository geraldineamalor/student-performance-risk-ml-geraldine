import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
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
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=500))
])
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
print("\nDetailed Classification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=encoder.classes_
))
accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)
print("Classification Model Performance (with Pipeline)")
print("Accuracy:", round(accuracy, 2))
print("Confusion Matrix:")
print(cm)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Student Risk Classification")
plt.savefig("assets/confusion_matrix.png", bbox_inches="tight")
plt.show()
with open("model/risk_model.pkl", "wb") as file:
    pickle.dump(pipeline, file)
with open("model/risk_label_encoder.pkl", "wb") as file:
    pickle.dump(encoder, file)

print("Pipeline model and label encoder saved successfully")

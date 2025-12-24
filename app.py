import streamlit as st
import pickle
import numpy as np
score_model = pickle.load(open("model/score_model.pkl", "rb"))
risk_model = pickle.load(open("model/risk_model.pkl", "rb"))
label_encoder = pickle.load(open("model/risk_label_encoder.pkl", "rb"))
st.title("Student Performance Risk Prediction")
st.write("Enter student details to predict performance score and risk level.")
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=75)
internal_marks = st.number_input("Internal Marks", min_value=0, max_value=100, value=70)
assignments = st.number_input("Assignments Submitted", min_value=0, max_value=10, value=5)
study_hours = st.number_input("Study Hours per Day", min_value=0, max_value=10, value=3)
if st.button("Predict"):
    features = np.array([[attendance, internal_marks, assignments, study_hours]])
    predicted_score = score_model.predict(features)[0]
    predicted_risk_encoded = risk_model.predict(features)[0]
    predicted_risk = label_encoder.inverse_transform([predicted_risk_encoded])[0]
    st.success(f"Predicted Performance Score: {predicted_score:.2f}")
    st.warning(f"Predicted Risk Level: {predicted_risk}")

import streamlit as st
import pickle
import pandas as pd
pipeline = pickle.load(open("model/risk_model.pkl", "rb"))
label_encoder = pickle.load(open("model/risk_label_encoder.pkl", "rb"))
st.title("Student Performance Risk Prediction")
st.write("Enter student details to predict academic risk level.")
attendance = st.number_input("Attendance (%)", 0, 100, 75)
internal_marks = st.number_input("Internal Marks", 0, 100, 70)
assignments = st.number_input("Assignments Submitted", 0, 10, 5)
study_hours = st.number_input("Study Hours per Day", 0, 10, 3)

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "Attendance": attendance,
        "InternalMarks": internal_marks,
        "Assignments": assignments,
        "StudyHours": study_hours
    }])
    proba = pipeline.predict_proba(input_data)[0]
    prediction_index = proba.argmax()
    prediction_label = label_encoder.inverse_transform([prediction_index])[0]
    confidence = proba[prediction_index] * 100
    st.success(f"Predicted Risk Level: {prediction_label}")
    st.info(f"Confidence: {confidence:.2f}%")

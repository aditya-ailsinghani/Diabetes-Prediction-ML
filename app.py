import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model components
model = joblib.load("diabetes_dt_model.pkl")
scaler = joblib.load("scaler.pkl")
selected_features = joblib.load("selected_features.pkl")

# Page setup
st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🧠 Diabetes Risk Prediction")
st.markdown("Enter medical details below to predict the probability of diabetes using a trained Decision Tree model.")

# Create input form
with st.form("prediction_form"):
    st.subheader("📋 Patient Details")

    # Create a dictionary to store user input
    user_input = {}

    if "Pregnancies" in selected_features:
        user_input["Pregnancies"] = st.number_input("Pregnancies", min_value=0, max_value=20, step=1, value=1)

    if "Glucose" in selected_features:
        user_input["Glucose"] = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, step=1, value=120)

    if "BloodPressure" in selected_features:
        user_input["BloodPressure"] = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, step=1, value=70)

    if "BMI" in selected_features:
        user_input["BMI"] = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1, value=28.0)

    if "DiabetesPedigreeFunction" in selected_features:
        user_input["DiabetesPedigreeFunction"] = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, step=0.01, value=0.5)

    submitted = st.form_submit_button("🔍 Predict")

# On submit
if submitted:
    # Convert input into dataframe
    input_df = pd.DataFrame([user_input])

    # Debug: show input
    st.subheader("📊 Input Summary")
    st.write(input_df)

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1] * 100

    # Output
    st.subheader("🎯 Prediction")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Diabetes\nConfidence: {prob:.2f}%")
    else:
        st.success(f"✅ Low Risk of Diabetes\nConfidence: {100 - prob:.2f}%")

    # Optional: show scaled input
    with st.expander("🔧 Scaled Input Data"):
        st.dataframe(pd.DataFrame(scaled_input, columns=selected_features))
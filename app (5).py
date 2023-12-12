import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load the trained model and the scaler
model = pickle.load(open('/content/model.pkl','rb'))  # Replace with the path to your model file
scaler = pickle.load(open('/content/minmax.pkl','rb'))  # Replace with the path to your scaler file

st.title("Heart Disease Prediction App")

st.sidebar.header("User Input Features")

# Create input fields for user to enter features
age = st.sidebar.slider("Age", 18, 100, 50)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
cp = st.sidebar.slider("Chest Pain Type", 0, 3, 1)
resting_bp = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
cholesterol = st.sidebar.slider("Cholesterol (mg/dL)", 100, 400, 200)
fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar (mg/dL)", ["< 120 mg/dL", ">= 120 mg/dL"])
resting_ecg = st.sidebar.selectbox("Resting ECG", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
max_hr = st.sidebar.slider("Maximum Heart Rate", 70, 220, 150)
exercise_angina = st.sidebar.selectbox("Exercise-Induced Angina", ["No", "Yes"])
oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", 0.0, 6.2, 1.0)
st_slope = st.sidebar.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

# Map categorical input to numerical values
sex = 0 if sex == "Male" else 1
fasting_bs = 0 if fasting_bs == "< 120 mg/dL" else 1

# Map Resting ECG values
if resting_ecg == "Normal":
    resting_ecg = 0
elif resting_ecg == "ST-T wave abnormality":
    resting_ecg = 1
else:
    resting_ecg = 2

# Map Exercise-Induced Angina
exercise_angina = 0 if exercise_angina == "No" else 1

# Map ST Slope
if st_slope == "Upsloping":
    st_slope = 0
elif st_slope == "Flat":
    st_slope = 1
else:
    st_slope = 2

# Create a feature vector from the user input
user_input = [age, sex, cp, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]

# Apply scaling to the user input
user_input = scaler.transform([user_input])

# Add a prediction button
if st.button("Predict"):
    # Make predictions using the model
    prediction = model.predict(user_input)
    st.subheader("Prediction")
    if prediction[0] == 1:
        st.write("The patient is at risk of heart disease.")
    else:
        st.write("The patient is not at risk of heart disease")

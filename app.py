import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model and scalers
# Make sure these files are in the same directory as your app.py
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
standard_scaler = pickle.load(open('standard_scaler.pkl', 'rb'))
robust_scaler = pickle.load(open('robust_scaler.pkl', 'rb'))

# Streamlit App Title
st.title('Hospital Clinical Deterioration Prediction')
st.write('Enter patient clinical data to predict deterioration in the next 12 hours.')

# Input fields for patient data
st.header('Patient Clinical Data')

# Numerical inputs
hour_from_admission = st.slider('Hour from Admission', 0, 71, 24)
heart_rate = st.number_input('Heart Rate (bpm)', 40.0, 180.0, 89.0)
respiratory_rate = st.number_input('Respiratory Rate (breaths/min)', 8.0, 45.0, 20.0)
spo2_pct = st.number_input('SpO2 (%)', 70.0, 100.0, 93.0)
temperature_c = st.number_input('Temperature (C)', 35.0, 40.0, 36.9)
systolic_bp = st.number_input('Systolic BP (mmHg)', 70.0, 185.0, 113.0)
diastolic_bp = st.number_input('Diastolic BP (mmHg)', 40.0, 110.0, 70.0)
oxygen_flow = st.number_input('Oxygen Flow (L/min)', 0.0, 60.0, 0.0)
mobility_score = st.slider('Mobility Score (0-4)', 0, 4, 2)
nurse_alert = st.selectbox('Nurse Alert (0=No, 1=Yes)', [0, 1])
wbc_count = st.number_input('WBC Count (x10^9/L)', 2.0, 30.0, 9.0)
lactate = st.number_input('Lactate (mmol/L)', 0.5, 8.0, 2.0)
creatinine = st.number_input('Creatinine (mg/dL)', 0.4, 4.5, 1.3)
crp_level = st.number_input('CRP Level (mg/L)', 0.0, 250.0, 34.0)
hemoglobin = st.number_input('Hemoglobin (g/dL)', 7.0, 17.0, 13.0)
sepsis_risk_score = st.number_input('Sepsis Risk Score', 0.0, 1.0, 0.5, format="%.4f")
age = st.slider('Age (years)', 18, 90, 50)
comorbidity_index = st.slider('Comorbidity Index (0-8)', 0, 8, 4)

# Categorical inputs
gender = st.selectbox('Gender', ['M', 'F'])
oxygen_device = st.selectbox('Oxygen Device', ['none', 'nasal', 'mask', 'hfnc', 'niv'])
admission_type = st.selectbox('Admission Type', ['Elective', 'Transfer', 'ED'])

# Create a DataFrame from inputs
input_data = pd.DataFrame({
    'hour_from_admission': [hour_from_admission],
    'heart_rate': [heart_rate],
    'respiratory_rate': [respiratory_rate],
    'spo2_pct': [spo2_pct],
    'temperature_c': [temperature_c],
    'systolic_bp': [systolic_bp],
    'diastolic_bp': [diastolic_bp],
    'oxygen_flow': [oxygen_flow],
    'mobility_score': [mobility_score],
    'nurse_alert': [nurse_alert],
    'wbc_count': [wbc_count],
    'lactate': [lactate],
    'creatinine': [creatinine],
    'crp_level': [crp_level],
    'hemoglobin': [hemoglobin],
    'sepsis_risk_score': [sepsis_risk_score],
    'age': [age],
    'comorbidity_index': [comorbidity_index],
    'gender': [gender],
    'oxygen_device': [oxygen_device],
    'admission_type': [admission_type]
})

# Feature Engineering (must match training data preprocessing)
input_data['mean_arterial_pressure'] = (input_data['systolic_bp'] + 2 * input_data['diastolic_bp']) / 3
input_data['pulse_pressure'] = input_data['systolic_bp'] - input_data['diastolic_bp']

# One-hot encode categorical features
input_data = pd.get_dummies(input_data, columns=['gender', 'oxygen_device', 'admission_type'], drop_first=False)

# Ensure all expected columns are present after one-hot encoding, fill missing with 0
# This list of columns should be generated from the X_train_hybrid dataframe used for training
# For demonstration, a predefined list is used. In a real app, load this from a saved list.
expected_columns = [
    'hour_from_admission', 'heart_rate', 'respiratory_rate', 'spo2_pct',
    'temperature_c', 'systolic_bp', 'diastolic_bp', 'oxygen_flow',
    'mobility_score', 'nurse_alert', 'wbc_count', 'lactate', 'creatinine',
    'crp_level', 'hemoglobin', 'sepsis_risk_score', 'age', 'comorbidity_index',
    'mean_arterial_pressure', 'pulse_pressure',
    'gender_F', 'gender_M',
    'oxygen_device_hfnc', 'oxygen_device_mask', 'oxygen_device_nasal', 'oxygen_device_niv', 'oxygen_device_none',
    'admission_type_ED', 'admission_type_Elective', 'admission_type_Transfer'
]

for col in expected_columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns to match the training data
input_data = input_data[expected_columns]

# Scale numerical features (must match training data preprocessing)
ss_cols=['heart_rate','respiratory_rate','temperature_c','wbc_count','creatinine','hemoglobin', 'mean_arterial_pressure', 'pulse_pressure']
rbt_cols=['spo2_pct','lactate','crp_level']

input_data[ss_cols] = standard_scaler.transform(input_data[ss_cols])
input_data[rbt_cols] = robust_scaler.transform(input_data[rbt_cols])

if st.button('Predict Deterioration'):
    prediction = rf_model.predict(input_data)
    prediction_proba = rf_model.predict_proba(input_data)[:, 1]

    if prediction[0] == 1:
        st.error(f'Prediction: High risk of deterioration in the next 12 hours (Probability: {prediction_proba[0]:.2f})')
    else:
        st.success(f'Prediction: Low risk of deterioration in the next 12 hours (Probability: {prediction_proba[0]:.2f})')

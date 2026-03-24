import streamlit as st
import pandas as pd # Ensure pandas is imported
import joblib # Ensure joblib is imported
from scipy.stats.mstats import winsorize # Added for winsorization

# Load the pre-trained model and scalers
rf_model = joblib.load('rf_model.pkl')
standard_scaler = joblib.load('standard_scaler.pkl')
robust_scaler = joblib.load('robust_scaler.pkl')

# Define the columns that were scaled by StandardScaler and RobustScaler
ss_cols = ['heart_rate', 'respiratory_rate', 'temperature_c', 'wbc_count', 'creatinine', 'hemoglobin']
rbt_cols = ['spo2_pct', 'lactate', 'crp_level']

# Define winsorization limits used during training
winsor_limits = {
    'heart_rate': [0.01, 0.05],
    'respiratory_rate': [0, 0.05],
    'spo2_pct': [0.1, 0],
    'temperature_c': [0.01, 0.05],
    'systolic_bp': [0.05, 0.05],
    'diastolic_bp': [0.05, 0.05],
    'oxygen_flow': [0.05, 0.05],
    'wbc_count': [0.01, 0.07],
    'lactate': [0, 0.15],
    'creatinine': [0, 0.07],
    'crp_level': [0, 0.1],
    'hemoglobin': [0.05, 0.01]
}

# The exact list of features the model was trained on, in the correct order.
# This list *does not* include 'mean_arterial_pressure' and 'pulse_pressure'
# because the model was trained before those features were added to the main dataframe (df1).
expected_features = ['hour_from_admission', 'heart_rate', 'respiratory_rate', 'spo2_pct',
                     'temperature_c', 'systolic_bp', 'diastolic_bp', 'oxygen_flow',
                     'mobility_score', 'nurse_alert', 'wbc_count', 'lactate',
                     'creatinine', 'crp_level', 'hemoglobin', 'sepsis_risk_score', 'age',
                     'comorbidity_index', 'oxygen_device_hfnc', 'oxygen_device_mask',
                     'oxygen_device_nasal', 'oxygen_device_niv', 'oxygen_device_none',
                     'gender_F', 'gender_M', 'admission_type_ED',
                     'admission_type_Elective', 'admission_type_Transfer']

st.title('Hospital Deterioration Prediction')
st.write('Enter patient details to predict the likelihood of deterioration within the next 12 hours.')

# Input widgets for numerical features
st.sidebar.header('Patient Clinical Data')
hour_from_admission = st.sidebar.number_input('Hours From Admission', min_value=0, max_value=71, value=24)
heart_rate = st.sidebar.number_input('Heart Rate', min_value=40.0, max_value=180.0, value=85.0, step=0.1)
respiratory_rate = st.sidebar.number_input('Respiratory Rate', min_value=8.0, max_value=45.0, value=20.0, step=0.1)
spo2_pct = st.sidebar.number_input('SpO2 Percentage', min_value=70.0, max_value=100.0, value=95.0, step=0.1)
temperature_c = st.sidebar.number_input('Temperature (C)', min_value=35.0, max_value=41.0, value=37.0, step=0.1)
systolic_bp = st.sidebar.number_input('Systolic BP', min_value=70.0, max_value=185.0, value=120.0, step=0.1)
diastolic_bp = st.sidebar.number_input('Diastolic BP', min_value=40.0, max_value=110.0, value=75.0, step=0.1)
oxygen_flow = st.sidebar.number_input('Oxygen Flow', min_value=0.0, max_value=60.0, value=0.0, step=0.1)
mobility_score = st.sidebar.slider('Mobility Score (0-4)', 0, 4, 2)
nurse_alert = st.sidebar.radio('Nurse Alert Triggered', [0, 1], index=0)
wbc_count = st.sidebar.number_input('WBC Count', min_value=2.0, max_value=30.0, value=9.0, step=0.1)
lactate = st.sidebar.number_input('Lactate', min_value=0.5, max_value=8.0, value=1.5, step=0.01)
creatinine = st.sidebar.number_input('Creatinine', min_value=0.4, max_value=4.5, value=1.0, step=0.01)
crp_level = st.sidebar.number_input('CRP Level', min_value=0.0, max_value=250.0, value=30.0, step=0.1)
hemoglobin = st.sidebar.number_input('Hemoglobin', min_value=7.0, max_value=17.0, value=13.0, step=0.1)
sepsis_risk_score = st.sidebar.number_input('Sepsis Risk Score', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
age = st.sidebar.number_input('Age', min_value=18, max_value=90, value=50)
comorbidity_index = st.sidebar.slider('Comorbidity Index (0-8)', 0, 8, 4)

# Input widgets for categorical features
st.sidebar.header('Patient Background')
oxygen_device_options = ['none', 'nasal', 'mask', 'hfnc', 'niv']
oxygen_device = st.sidebar.selectbox('Oxygen Device', oxygen_device_options)
gender_options = ['M', 'F']
gender = st.sidebar.selectbox('Gender', gender_options)
admission_type_options = ['Elective', 'Transfer', 'ED']
admission_type = st.sidebar.selectbox('Admission Type', admission_type_options)

if st.button('Predict Deterioration'):
    # Collect inputs into a DataFrame
    input_data = {
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
        'oxygen_device': [oxygen_device],
        'gender': [gender],
        'admission_type': [admission_type]
    }
    input_df = pd.DataFrame(input_data)

    # Apply Winsorization to input_df using the defined limits
    for col, limits in winsor_limits.items():
        if col in input_df.columns:
            input_df[col] = winsorize(input_df[col].values, limits=limits)

    # Apply one-hot encoding for categorical features
    input_df = pd.get_dummies(input_df, columns=['oxygen_device', 'gender', 'admission_type'])

    # Reindex input_df to match the expected_features list and fill any missing columns with 0
    # This ensures consistency in columns and their order for the model.
    input_df = input_df.reindex(columns=expected_features, fill_value=0)

    # Convert boolean columns to integer (0 or 1) that might result from get_dummies for new data
    bool_cols = input_df.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        input_df[bool_cols] = input_df[bool_cols].astype(int)

    # Apply StandardScaler
    input_df[ss_cols] = standard_scaler.transform(input_df[ss_cols])

    # Apply RobustScaler
    input_df[rbt_cols] = robust_scaler.transform(input_df[rbt_cols])

    # Make prediction
    prediction = rf_model.predict(input_df)
    prediction_proba = rf_model.predict_proba(input_df)
        
    st.subheader('Prediction Result:')

    print(f"Prediction for new patient: {prediction[0]}")
    print(f"Probability of no deterioration (0): {prediction_proba[0][0]:.4f}")
    print(f"Probability of deterioration (1): {prediction_proba[0][1]:.4f}")

    if prediction[0] == 1:
        st.error(f"The new patient is predicted to deteriorate within the next 12 hours.")
    else:
        st.success(f"The new patient is predicted not to deteriorate within the next 12 hours.")
        st.write('Disclaimer: This prediction is for informational purposes only and should not replace professional medical advice.')

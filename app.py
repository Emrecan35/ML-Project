import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("catboost_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Water Potability Prediction", layout="centered")
st.title("💧 Water Potability Prediction")

st.write("Enter the following water test results:")

# Input fields for each feature
ph = st.sidebar.slider("pH", 4.0, 10.0, 7.0, step=0.1)
hardness = st.number_input("Hardness", min_value=0.0, value=150.0)
solids = st.number_input("Solids", min_value=0.0, value=10000.0)
chloramines = st.number_input("Chloramines", min_value=0.0, value=7.0)
sulfate = st.number_input("Sulfate", min_value=0.0, value=330.0)
conductivity = st.number_input("Conductivity", min_value=0.0, value=400.0)
organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=10.0)
trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=60.0)
turbidity = st.number_input("Turbidity", min_value=0.0, value=3.5)

# Prepare input
user_input = np.array([[ph, hardness, solids, chloramines, sulfate,
                        conductivity, organic_carbon, trihalomethanes, turbidity]])

# Scale input
user_input_scaled = scaler.transform(user_input)

# Predict button
if st.button("🔎 Predict Potability"):
    prediction = model.predict(user_input_scaled)[0]
    probability = model.predict_proba(user_input_scaled)[0][1]

    if prediction == 1:
        st.success(f"✅ This water is likely **POTABLE**.\n\nConfidence: {probability:.2%}")
    else:
        st.error(f"⚠️ This water is likely **NOT POTABLE**.\n\nConfidence: {probability:.2%}")

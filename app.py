import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Dosya yolları (kendi dosya isimlerine göre düzenle)
MODEL_PATH = "catboost_model.pkl"
SCALER_PATH = "scaler.pkl"
DEFAULTS_PATH = "impute_defaults.pkl"

st.set_page_config(page_title="Water Potability Prediction", page_icon="💧", layout="wide")

@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    defaults = joblib.load(DEFAULTS_PATH)
    return model, scaler, defaults

def get_user_input(defaults):
    st.sidebar.header("Input Water Quality Features")

    # Default değerleri kullanarak slider oluşturuyoruz
    ph = st.sidebar.slider("pH", 0.0, 14.0, float(defaults.get("ph", 7.0)), step=0.1)
    hardness = st.sidebar.slider("Hardness", 0.0, 500.0, float(defaults.get("Hardness", 150.0)), step=1.0)
    solids = st.sidebar.slider("Solids (ppm)", 0.0, 50000.0, float(defaults.get("Solids", 20000.0)), step=10.0)
    chloramines = st.sidebar.slider("Chloramines", 0.0, 20.0, float(defaults.get("Chloramines", 7.0)), step=0.1)
    sulfate = st.sidebar.slider("Sulfate", 0.0, 500.0, float(defaults.get("Sulfate", 250.0)), step=1.0)
    conductivity = st.sidebar.slider("Conductivity", 0.0, 1500.0, float(defaults.get("Conductivity", 300.0)), step=1.0)
    organic_carbon = st.sidebar.slider("Organic Carbon", 0.0, 20.0, float(defaults.get("Organic_carbon", 5.0)), step=0.1)
    trihalomethanes = st.sidebar.slider("Trihalomethanes", 0.0, 150.0, float(defaults.get("Trihalomethanes", 40.0)), step=0.1)
    turbidity = st.sidebar.slider("Turbidity", 0.0, 15.0, float(defaults.get("Turbidity", 3.0)), step=0.1)

    data = {
        "ph": ph,
        "Hardness": hardness,
        "Solids": solids,
        "Chloramines": chloramines,
        "Sulfate": sulfate,
        "Conductivity": conductivity,
        "Organic_carbon": organic_carbon,
        "Trihalomethanes": trihalomethanes,
        "Turbidity": turbidity
    }

    input_df = pd.DataFrame([data])

    # Eksik değer doldurma (örneğin kullanıcı boş bırakırsa)
    for col in ["ph", "Sulfate", "Trihalomethanes"]:
        if input_df[col].isnull().any():
            input_df[col] = input_df[col].fillna(defaults.get(col, 0))

    return input_df

def main():
    st.title("💧 Water Potability Prediction App")
    st.write("CatBoost model kullanılarak su içilebilirliği tahmini yapılmaktadır.")

    # Model, scaler ve default değerler yükleniyor
    model, scaler, defaults = load_resources()

    # Kullanıcıdan veri alınıyor
    input_df = get_user_input(defaults)

    st.subheader("Girdiğiniz Özellikler")
    st.write(input_df)

    # Ölçekleme
    input_scaled = scaler.transform(input_df)

    if st.button("Tahmin Et"):
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.success(f"✅ Bu su İÇİLEBİLİR.\n\nGüven Skoru: %{probability*100:.2f}")
        else:
            st.error(f"⚠️ Bu su İÇİLEMEZ.\n\nGüven Skoru: %{probability*100:.2f}")

if __name__ == "__main__":
    main()

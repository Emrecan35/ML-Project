import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Dosya yolları
MODEL_PATH = "catboost_model.pkl"
SCALER_PATH = "scaler.pkl"
IMPUTE_DEFAULTS_PATH = "impute_defaults.pkl"

# Sayfa ayarları
st.set_page_config(
    page_title="💧 Water Potability Prediction",
    page_icon="💧",
    layout="centered"
)

# Model, scaler ve defaults yükle
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    defaults = joblib.load(IMPUTE_DEFAULTS_PATH)
    return model, scaler, defaults

def get_user_input(defaults):
    st.sidebar.title("🔧 Su Kalitesi Girdileri")

    ph = st.sidebar.slider("pH", 0.0, 14.0, 7.0, step=0.1)
    hardness = st.sidebar.slider("Hardness", 0.0, 500.0, 150.0, step=1.0)
    solids = st.sidebar.slider("Solids (ppm)", 0.0, 50000.0, 20000.0, step=10.0)
    chloramines = st.sidebar.slider("Chloramines", 0.0, 20.0, 7.0, step=0.1)
    sulfate = st.sidebar.slider("Sulfate", 0.0, 500.0, 250.0, step=1.0)
    conductivity = st.sidebar.slider("Conductivity", 0.0, 1500.0, 300.0, step=1.0)
    organic_carbon = st.sidebar.slider("Organic Carbon", 0.0, 20.0, 5.0, step=0.1)
    trihalomethanes = st.sidebar.slider("Trihalomethanes", 0.0, 150.0, 40.0, step=0.1)
    turbidity = st.sidebar.slider("Turbidity", 0.0, 15.0, 3.0, step=0.1)

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
    input_df.fillna(defaults, inplace=True)
    return input_df

def main():
    st.title("💧 Water Potability Prediction App")
    st.markdown("""
    Bu uygulama, verilen su kalitesi parametrelerine göre **CatBoost** modeli ile suyun içilebilirliğini tahmin eder.
    
    Girdileri sol menüden ayarlayabilir, ardından tahmin butonuna basarak sonucu görebilirsiniz.
    """)

    model, scaler, defaults = load_model_and_scaler()
    input_df = get_user_input(defaults)

    st.subheader("📥 Girdiğiniz Değerler")
    st.dataframe(input_df)

    input_scaled = scaler.transform(input_df)

    if st.button("🔍 Tahmin Et"):
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.success("✅ Tahmin: Bu su **İÇİLEBİLİR**.")
        else:
            st.error("❌ Tahmin: Bu su **İÇİLEMEZ**.")

        st.info(f"💡 Güven Oranı: {probability:.1%}")

if __name__ == "__main__":
    main()

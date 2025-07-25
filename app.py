import streamlit as st
import pandas as pd
import joblib
from prophet import Prophet
import datetime

# ===============================
# Load Model
# ===============================
rf_model = joblib.load("rf_esg_model.pkl")
prophet_models = joblib.load("all_prophet_models.pkl")  # contains 'depth', 'ph', 'tds'

# ===============================
# Load Data
# ===============================
data = pd.read_csv("pre-processingdata.csv")
data = data[['Date', 'Depth', 'pH', 'TDS', 'Location']]
data.columns = ['ds', 'depth', 'ph', 'tds', 'location']
data['ds'] = pd.to_datetime(data['ds'])
data['location_encoded'] = data['location'].astype('category').cat.codes

# ===============================
# App Title
# ===============================
st.title("📊 Prediksi & Klasifikasi Risiko Limbah Radioaktif")
st.markdown("""
#### Pilih tanggal prediksi (tahun >= 2020)
Aplikasi ini akan memprediksi *depth*, *pH*, dan *TDS* dari data historis,
lalu mengklasifikasikan potensi risikonya (Rendah / Sedang / Tinggi).
""")

# ===============================
# Date Input
# ===============================
selected_date = st.date_input("Pilih tanggal prediksi", value=datetime.date(2025, 1, 1), min_value=datetime.date(2020, 1, 1))

if st.button("🔍 Prediksi & Klasifikasi"):
    with st.spinner("Sedang memproses..."):

        # Format future dataframe
        future = pd.DataFrame({"ds": [pd.to_datetime(selected_date)]})
        future['location_encoded'] = [data['location_encoded'].iloc[-1]]

        # Forecast per fitur
        def get_forecast(model, df):
            prediction = model.predict(df)
            return prediction['yhat'].values[0]

        depth_model = prophet_models['depth']
        ph_model = prophet_models['ph']
        tds_model = prophet_models['tds']

        pred_depth = min(get_forecast(depth_model, future), 20)  # limit max
        pred_ph = max(get_forecast(ph_model, future), 0)         # limit min
        pred_tds = get_forecast(tds_model, future)

        # Tampilkan hasil forecast
        st.subheader("🔮 Hasil Forecast:")
        st.write(f"**Depth**: {pred_depth:.2f}")
        st.write(f"**pH**: {pred_ph:.2f}")
        st.write(f"**TDS**: {pred_tds:.2f}")

        # Klasifikasi
        input_df = pd.DataFrame([[pred_depth, pred_ph, pred_tds]], columns=["depth", "ph", "tds"])
        risk_pred = rf_model.predict(input_df)[0]

        st.subheader("🧪 Hasil Klasifikasi Risiko:")
        st.success(f"Prediksi ESG Risk: {risk_pred}")

st.markdown("---")
st.caption("Model by: Salsa | NUCLIFY")
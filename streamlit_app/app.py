import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="AQI Forecasting Platform",
    layout="wide"
)
st.info("🔹 Demo Mode: Using precomputed AQI predictions")

st.title("🌍 Air Quality Index (AQI) Forecasting")
st.markdown("AI-driven AQI visualization and forecasting dashboard")

# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource
def load_scaler():
    return joblib.load("models/scaler.pkl")

@st.cache_data
def load_predictions():
    data = np.load("data/predictions.npz")
    return data["preds"]

scaler = load_scaler()

try:
    predictions = load_predictions()
    st.success("Predictions loaded successfully")
except:
    st.warning("Predictions file not found. Using demo data.")
    predictions = np.random.randint(50, 300, 30)

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("🔧 Simulation Inputs")

city = st.sidebar.selectbox(
    "Select City",
    ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata"]
)

days = st.sidebar.slider("Forecast Days", 7, 30, 14)

# -----------------------------
# DataFrame
# -----------------------------
df = pd.DataFrame({
    "Day": np.arange(1, days + 1),
    "Predicted AQI": predictions[:days]
})

# -----------------------------
# Plot
# -----------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["Day"],
    y=df["Predicted AQI"],
    mode="lines+markers",
    name="AQI"
))

fig.update_layout(
    title=f"AQI Forecast for {city}",
    xaxis_title="Day",
    yaxis_title="AQI",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# AQI interpretation
# -----------------------------
def aqi_label(aqi):
    if aqi <= 50:
        return "Good 🟢"
    elif aqi <= 100:
        return "Satisfactory 🟡"
    elif aqi <= 200:
        return "Moderate 🟠"
    elif aqi <= 300:
        return "Poor 🔴"
    else:
        return "Severe 🟣"

st.subheader("📊 AQI Classification")
df["AQI Category"] = df["Predicted AQI"].apply(aqi_label)
st.dataframe(df)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "Built by **Vedika Rana** | AI & Data Science | AQI Forecasting Platform"
)


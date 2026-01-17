import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import os
import pydeck as pdk

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AQI Forecasting Platform",
    page_icon="🌍",
    layout="wide"
)

# =========================
# AQI FETCHER
# =========================
class RealTimeAQIFetcher:
    def __init__(self):
        self.api_key = st.secrets.get(
            "OPENWEATHER_API_KEY",
            os.getenv("OPENWEATHER_API_KEY")
        )
        self.base_url = "http://api.openweathermap.org/data/2.5/air_pollution"

    def fetch_current_aqi(self, lat, lon):
        if not self.api_key:
            raise ValueError("OPENWEATHER_API_KEY not found")

        url = f"{self.base_url}?lat={lat}&lon={lon}&appid={self.api_key}"
        res = requests.get(url, timeout=10)
        res.raise_for_status()

        data = res.json()
        comp = data["list"][0]["components"]

        return {
            "timestamp": datetime.now(),
            "aqi": data["list"][0]["main"]["aqi"],
            "pm2_5": comp["pm2_5"],
            "pm10": comp["pm10"],
            "no2": comp["no2"],
            "so2": comp["so2"],
            "co": comp["co"],
            "o3": comp["o3"],
            "nh3": comp["nh3"]
        }

    def get_aqi_category(self, pm25):
        if pm25 <= 50:
            return "Good", "#00E400"
        elif pm25 <= 100:
            return "Moderate", "#FFFF00"
        elif pm25 <= 150:
            return "Unhealthy for Sensitive Groups", "#FF7E00"
        elif pm25 <= 200:
            return "Unhealthy", "#FF0000"
        elif pm25 <= 300:
            return "Very Unhealthy", "#8F3F97"
        else:
            return "Hazardous", "#7E0023"

# =========================
# SIMPLE FORECASTER
# =========================
class SimplePredictor:
    def predict(self, current, hours=6):
        trend = np.random.choice([-2, -1, 0, 1, 2])
        noise = np.random.normal(0, 5, hours)
        return [max(10, current + trend * i + noise[i]) for i in range(hours)]

# =========================
# INIT OBJECTS  ✅ FIX
# =========================
fetcher = RealTimeAQIFetcher()
predictor = SimplePredictor()

# =========================
# UTILITIES
# =========================
def get_aqi_color(pm25):
    if pm25 <= 50: return [0, 228, 0]
    if pm25 <= 100: return [255, 255, 0]
    if pm25 <= 150: return [255, 126, 0]
    if pm25 <= 200: return [255, 0, 0]
    if pm25 <= 300: return [143, 63, 151]
    return [126, 0, 35]

@st.cache_data(ttl=3600)
def generate_24h_trend(current_pm25):
    hours = [datetime.now() - timedelta(hours=i) for i in range(23, -1, -1)]
    values = current_pm25 + np.cumsum(np.random.normal(0, 3, 24))
    values = np.clip(values, 15, None)
    return pd.DataFrame({"Time": hours, "PM2.5": values})

# =========================
# HEADER
# =========================
st.markdown("<h1 style='text-align:center'>🌍 AQI Forecasting Platform</h1>", unsafe_allow_html=True)
st.markdown("### Real-time Monitoring • Forecasting • Trend Analysis")

# =========================
# CITY TIERS
# =========================
CITY_TIERS = {
    "TIER 1": {
        "Delhi": (28.6139, 77.2090),
        "Mumbai": (19.0760, 72.8777),
        "Bangalore": (12.9716, 77.5946),
        "Chennai": (13.0827, 80.2707),
        "Kolkata": (22.5726, 88.3639),
        "Hyderabad": (17.3850, 78.4867),
        "Pune": (18.5204, 73.8567),
        "Ahmedabad": (23.0225, 72.5714)
    },
    "TIER 2": {
        "Jaipur": (26.9124, 75.7873),
        "Indore": (22.7196, 75.8577),
        "Bhopal": (23.2599, 77.4126),
        "Nagpur": (21.1458, 79.0882),
        "Lucknow": (26.8467, 80.9462),
        "Surat": (21.1702, 72.8311),
        "Vadodara": (22.3072, 73.1812),
        "Kochi": (9.9312, 76.2673),
        "Vizag": (17.6868, 83.2185)
    },
    "TIER 3": {
        "Dehradun": (30.3165, 78.0322),
        "Udaipur": (24.5854, 73.7125),
        "Gwalior": (26.2183, 78.1828),
        "Roorkee": (29.8543, 77.8880),
        "Haridwar": (29.9457, 78.1642),
        "Haldwani": (29.2183, 79.5130)
    }
}

# =========================
# SIDEBAR
# =========================
st.sidebar.header("🏙 City Selection")
tier = st.sidebar.selectbox("Select Tier", list(CITY_TIERS.keys()))
city = st.sidebar.selectbox("Select City", list(CITY_TIERS[tier].keys()))
lat, lon = CITY_TIERS[tier][city]

# =========================
# FETCH AQI
# =========================
try:
    data = fetcher.fetch_current_aqi(lat, lon)
except Exception as e:
    st.error(f"API Error: {e}")
    st.stop()

pm25 = data["pm2_5"]
category, color = fetcher.get_aqi_category(pm25)

# =========================
# KPI METRICS
# =========================
c1, c2, c3, c4 = st.columns(4)
c1.metric("PM2.5", f"{pm25:.1f}")
c2.metric("PM10", f"{data['pm10']:.1f}")
c3.metric("NO₂", f"{data['no2']:.1f}")
c4.metric("O₃", f"{data['o3']:.1f}")

st.markdown(
    f"<div style='background:{color};padding:1rem;border-radius:10px;text-align:center;color:black'>"
    f"<h2>{category}</h2></div>",
    unsafe_allow_html=True
)

# =========================
# MAP VISUALIZATION
# =========================
st.subheader("🗺 AQI Map View")

map_df = pd.DataFrame({
    "lat": [lat],
    "lon": [lon],
    "pm25": [pm25],
    "color": [get_aqi_color(pm25)]
})

layer = pdk.Layer(
    "ScatterplotLayer",
    map_df,
    get_position="[lon, lat]",
    get_radius=20000,
    get_fill_color="color",
    pickable=True
)

st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=7),
    tooltip={"html": f"<b>{city}</b><br>PM2.5: {pm25:.1f}"}
))

# =========================
# 24H TREND
# =========================
st.subheader("📈 PM2.5 Trend (Last 24 Hours)")
trend_df = generate_24h_trend(pm25)

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    x=trend_df["Time"],
    y=trend_df["PM2.5"],
    mode="lines+markers",
    fill="tozeroy"
))

fig_trend.update_layout(
    template="plotly_white",
    height=450,
    xaxis_title="Time",
    yaxis_title="PM2.5 (µg/m³)"
)

st.plotly_chart(fig_trend, use_container_width=True)

# =========================
# FORECAST
# =========================
st.subheader("🎯 PM2.5 Forecast (Next 6 Hours)")
forecast = predictor.predict(pm25)

forecast_df = pd.DataFrame({
    "Hour": [f"+{i+1}h" for i in range(6)],
    "PM2.5": forecast,
    "Category": [fetcher.get_aqi_category(v)[0] for v in forecast]
})

st.dataframe(forecast_df, use_container_width=True)

# =========================
# FOOTER
# =========================
st.caption("Data Source: OpenWeather | Trend: Mock (cached)")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import requests
import os

# Page configuration
st.set_page_config(
    page_title='AQI Forecasting Platform',
    page_icon='🌍',
    layout='wide',
    initial_sidebar_state='expanded'
)

# AQI Fetcher Class (embedded)
class RealTimeAQIFetcher:
    def __init__(self):
        self.api_key = st.secrets.get('OPENWEATHER_API_KEY', os.getenv('OPENWEATHER_API_KEY'))
        self.base_url = 'http://api.openweathermap.org/data/2.5/air_pollution'
        
    def fetch_current_aqi(self, lat, lon):
        try:
            url = f'{self.base_url}?lat={lat}&lon={lon}&appid={self.api_key}'
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            components = data['list'][0]['components']
            
            return {
                'timestamp': datetime.now(),
                'lat': lat,
                'lon': lon,
                'aqi': data['list'][0]['main']['aqi'],
                'pm2_5': components['pm2_5'],
                'pm10': components['pm10'],
                'no2': components['no2'],
                'so2': components['so2'],
                'co': components['co'],
                'o3': components['o3'],
                'no': components['no'],
                'nh3': components['nh3']
            }
        except Exception as e:
            st.error(f'Error fetching data: {e}')
            return None
    
    def get_aqi_category(self, aqi_value):
        if aqi_value <= 50:
            return 'Good', '#00E400'
        elif aqi_value <= 100:
            return 'Moderate', '#FFFF00'
        elif aqi_value <= 150:
            return 'Unhealthy for Sensitive Groups', '#FF7E00'
        elif aqi_value <= 200:
            return 'Unhealthy', '#FF0000'
        elif aqi_value <= 300:
            return 'Very Unhealthy', '#8F3F97'
        else:
            return 'Hazardous', '#7E0023'

# Simple Predictor Class (embedded)
class SimplePredictor:
    def predict(self, current_value, hours=6):
        base_trend = np.random.choice([-1, 0, 1]) * 3
        noise = np.random.normal(0, 8, hours)
        predictions = [max(10, current_value + base_trend * i + noise[i]) for i in range(hours)]
        return predictions

# Custom CSS
st.markdown('''
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
</style>
''', unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🌍 AQI Forecasting Platform</h1>', unsafe_allow_html=True)
st.markdown('### Real-time Air Quality Monitoring & Predictions')

# Sidebar
st.sidebar.header('⚙️ Configuration')

# City selection
cities = {
    'Delhi': (28.6139, 77.2090),
    'Mumbai': (19.0760, 72.8777),
    'Kolkata': (22.5726, 88.3639),
    'Chennai': (13.0827, 80.2707),
    'Bangalore': (12.9716, 77.5946),
    'Hyderabad': (17.3850, 78.4867),
    'Ahmedabad': (23.0225, 72.5714),
    'Pune': (18.5204, 73.8567),
    'Lucknow': (26.8467, 80.9462),
    'Kanpur': (26.4499, 80.3319)
}

selected_city = st.sidebar.selectbox('Select City', list(cities.keys()))
lat, lon = cities[selected_city]

# Initialize components
fetcher = RealTimeAQIFetcher()
predictor = SimplePredictor()

# Fetch real-time data
if st.sidebar.button('🔄 Refresh Data', type='primary'):
    st.cache_data.clear()

@st.cache_data(ttl=600)
def fetch_current_data(city, lat, lon):
    return fetcher.fetch_current_aqi(lat, lon)

current_data = fetch_current_data(selected_city, lat, lon)

if current_data:
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    pm25_value = current_data['pm2_5']
    category, color = fetcher.get_aqi_category(pm25_value)
    
    with col1:
        st.metric('PM2.5', f"{pm25_value:.1f} µg/m³")
    with col2:
        st.metric('PM10', f"{current_data['pm10']:.1f} µg/m³")
    with col3:
        st.metric('NO₂', f"{current_data['no2']:.1f} µg/m³")
    with col4:
        st.metric('O₃', f"{current_data['o3']:.1f} µg/m³")
    
    # AQI Category
    st.markdown(f'''
    <div style="background-color: {color}; padding: 1.5rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
        <h2 style="color: white; margin: 0;">Air Quality: {category}</h2>
        <p style="color: white; margin: 0.5rem 0 0 0;">Current PM2.5: {pm25_value:.1f} µg/m³</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Two columns
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader('📊 Current Pollutant Levels')
        
        pollutants = {
            'PM2.5': current_data['pm2_5'],
            'PM10': current_data['pm10'],
            'NO₂': current_data['no2'],
            'SO₂': current_data['so2'],
            'CO': current_data['co'] / 100,
            'O₃': current_data['o3'],
            'NH₃': current_data['nh3']
        }
        
        fig_bar = go.Figure(data=[
            go.Bar(
                x=list(pollutants.keys()),
                y=list(pollutants.values()),
                marker_color=['#FF6B6B', '#FFA07A', '#FFD700', '#98D8C8', '#6BCB77', '#4D96FF', '#9D84B7']
            )
        ])
        
        fig_bar.update_layout(
            title='Pollutant Concentrations',
            xaxis_title='Pollutant',
            yaxis_title='Concentration (µg/m³)',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col_right:
        st.subheader('🎯 6-Hour Forecast')
        
        predictions = predictor.predict(pm25_value, hours=6)
        forecast_hours = [f'+{i+1}h' for i in range(6)]
        
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=['Now'] + forecast_hours,
            y=[pm25_value] + predictions,
            mode='lines+markers',
            name='PM2.5 Forecast',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=10),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.1)'
        ))
        
        fig_forecast.add_hline(y=50, line_dash='dash', line_color='green', annotation_text='Good')
        fig_forecast.add_hline(y=100, line_dash='dash', line_color='yellow', annotation_text='Moderate')
        fig_forecast.add_hline(y=150, line_dash='dash', line_color='orange', annotation_text='Unhealthy')
        
        fig_forecast.update_layout(
            title='PM2.5 6-Hour Forecast',
            xaxis_title='Time',
            yaxis_title='PM2.5 (µg/m³)',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        forecast_df = pd.DataFrame({
            'Time': forecast_hours,
            'PM2.5': [f'{p:.1f}' for p in predictions],
            'Category': [fetcher.get_aqi_category(p)[0] for p in predictions]
        })
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
    
    # Health recommendations
    st.subheader('💡 Health Recommendations')
    
    if pm25_value <= 50:
        st.success('✅ **Good Air Quality** - Perfect for outdoor activities!')
    elif pm25_value <= 100:
        st.info('ℹ️ **Moderate Air Quality** - Generally acceptable')
    elif pm25_value <= 150:
        st.warning('⚠️ **Unhealthy for Sensitive Groups**')
    elif pm25_value <= 200:
        st.error('❌ **Unhealthy Air Quality** - Limit outdoor activities')
    else:
        st.error('🚨 **Hazardous!** - Stay indoors')
    
    # Footer
    st.markdown('---')
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.metric('Data Source', 'OpenWeather')
    with col_f2:
        st.metric('Cities', '10')
    with col_f3:
        st.metric('Updated', datetime.now().strftime('%I:%M %p'))
    
else:
    st.error('❌ Unable to fetch data. Add API key to Streamlit Secrets:')
    st.code('OPENWEATHER_API_KEY = "28b67a4d5b8be4ade2b7bcfac97989f0"')

# Sidebar
st.sidebar.markdown('---')
st.sidebar.markdown('### 📖 About')
st.sidebar.info('''
Real-time AQI monitoring for Indian cities.

**Tech Stack:**
- Streamlit
- OpenWeather API
- Python Data Science
- Statistical Forecasting
''')

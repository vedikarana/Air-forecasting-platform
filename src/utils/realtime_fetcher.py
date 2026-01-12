import requests
import pandas as pd
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

class RealTimeAQIFetcher:
    def __init__(self):
        self.api_key = os.getenv('OPENWEATHER_API_KEY')
        self.base_url = 'http://api.openweathermap.org/data/2.5/air_pollution'
        
    def fetch_current_aqi(self, lat, lon):
        '''Fetch current AQI data'''
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
            print(f'Error fetching AQI: {e}')
            return None
    
    def get_aqi_category(self, aqi_value):
        '''Get AQI category and color'''
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

import requests
import pandas as pd
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time
import json

load_dotenv()

class OpenWeatherCollector:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5/air_pollution"
        
    def get_current_aqi(self, lat, lon):
        '''Fetch current AQI data for a location'''
        url = f'{self.base_url}?lat={lat}&lon={lon}&appid={self.api_key}'
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Extract relevant data
            result = {
                'timestamp': datetime.fromtimestamp(data['list'][0]['dt']),
                'lat': lat,
                'lon': lon,
                'aqi': data['list'][0]['main']['aqi'],
                'co': data['list'][0]['components']['co'],
                'no': data['list'][0]['components']['no'],
                'no2': data['list'][0]['components']['no2'],
                'o3': data['list'][0]['components']['o3'],
                'so2': data['list'][0]['components']['so2'],
                'pm2_5': data['list'][0]['components']['pm2_5'],
                'pm10': data['list'][0]['components']['pm10'],
                'nh3': data['list'][0]['components']['nh3']
            }
            return result
        except Exception as e:
            print(f'Error fetching data: {e}')
            return None
    
    def get_historical_aqi(self, lat, lon, start_date, end_date):
        '''Fetch historical AQI data'''
        url = f'{self.base_url}/history'
        
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        params = {
            'lat': lat,
            'lon': lon,
            'start': start_timestamp,
            'end': end_timestamp,
            'appid': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            records = []
            for item in data['list']:
                record = {
                    'timestamp': datetime.fromtimestamp(item['dt']),
                    'lat': lat,
                    'lon': lon,
                    'aqi': item['main']['aqi'],
                    'co': item['components']['co'],
                    'no': item['components']['no'],
                    'no2': item['components']['no2'],
                    'o3': item['components']['o3'],
                    'so2': item['components']['so2'],
                    'pm2_5': item['components']['pm2_5'],
                    'pm10': item['components']['pm10'],
                    'nh3': item['components']['nh3']
                }
                records.append(record)
            
            return pd.DataFrame(records)
        except Exception as e:
            print(f'Error fetching historical data: {e}')
            return None
    
    def collect_multiple_cities(self, cities_coords, days_back=365):
        '''Collect data for multiple cities'''
        all_data = []
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        for city_name, (lat, lon) in cities_coords.items():
            print(f'Collecting data for {city_name}...')
            df = self.get_historical_aqi(lat, lon, start_date, end_date)
            
            if df is not None:
                df['city'] = city_name
                all_data.append(df)
                time.sleep(1)  # Rate limiting
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            return combined_df
        return None

if __name__ == '__main__':
    # Major Indian cities with high pollution
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
    
    collector = OpenWeatherCollector()
    
    # Collect historical data
    print('Collecting historical data...')
    df = collector.collect_multiple_cities(cities, days_back=365)
    
    if df is not None:
        # Save to CSV
        output_path = 'data/raw/aqi_historical_data.csv'
        df.to_csv(output_path, index=False)
        print(f'Data saved to {output_path}')
        print(f'Total records: {len(df)}')
        print(f'\nData preview:\n{df.head()}')
        print(f'\nData info:\n{df.info()}')
    else:
        print('Failed to collect data')

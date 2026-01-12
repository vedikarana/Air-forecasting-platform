import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_aqi_data(city_name, lat, lon, days=730, base_aqi=150):
    '''Generate realistic AQI time series data'''
    
    dates = pd.date_range(
        end=datetime.now(),
        periods=days * 24,  # Hourly data
        freq='H'
    )
    
    # Seasonal pattern (worse in winter)
    seasonal = 30 * np.sin(2 * np.pi * np.arange(len(dates)) / (365 * 24) - np.pi/2)
    
    # Weekly pattern (worse on weekdays)
    weekly = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / (7 * 24))
    
    # Daily pattern (worse in morning/evening rush hours)
    daily = 15 * (np.sin(2 * np.pi * np.arange(len(dates)) / 24 - np.pi/4) + 
                  np.sin(2 * np.pi * np.arange(len(dates)) / 24 - 5*np.pi/4))
    
    # Random noise
    noise = np.random.normal(0, 20, len(dates))
    
    # Combine patterns
    pm2_5 = np.maximum(10, base_aqi + seasonal + weekly + daily + noise)
    
    # Generate correlated pollutants
    pm10 = pm2_5 * 1.5 + np.random.normal(0, 10, len(dates))
    no2 = pm2_5 * 0.3 + np.random.normal(20, 5, len(dates))
    so2 = pm2_5 * 0.2 + np.random.normal(10, 3, len(dates))
    co = pm2_5 * 5 + np.random.normal(500, 100, len(dates))
    o3 = np.maximum(0, 50 - pm2_5 * 0.1 + np.random.normal(0, 10, len(dates)))
    no = no2 * 0.3 + np.random.normal(5, 2, len(dates))
    nh3 = pm2_5 * 0.1 + np.random.normal(5, 2, len(dates))
    
    # Calculate AQI based on PM2.5
    aqi = np.digitize(pm2_5, bins=[0, 50, 100, 150, 200, 300]) 
    
    # Add weather features
    temp = 25 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / (365 * 24)) + np.random.normal(0, 3, len(dates))
    humidity = 60 + 20 * np.sin(2 * np.pi * np.arange(len(dates)) / (365 * 24) + np.pi) + np.random.normal(0, 10, len(dates))
    wind_speed = np.maximum(0, 10 + np.random.normal(0, 5, len(dates)))
    pressure = 1013 + np.random.normal(0, 10, len(dates))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'city': city_name,
        'lat': lat,
        'lon': lon,
        'aqi': aqi,
        'pm2_5': pm2_5,
        'pm10': pm10,
        'no2': no2,
        'so2': so2,
        'co': co,
        'o3': o3,
        'no': no,
        'nh3': nh3,
        'temperature': temp,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'pressure': pressure
    })
    
    return df

if __name__ == '__main__':
    cities = {
        'Delhi': (28.6139, 77.2090, 200),
        'Mumbai': (19.0760, 72.8777, 120),
        'Kolkata': (22.5726, 88.3639, 150),
        'Chennai': (13.0827, 80.2707, 80),
        'Bangalore': (12.9716, 77.5946, 90),
        'Hyderabad': (17.3850, 78.4867, 100),
        'Ahmedabad': (23.0225, 72.5714, 130),
        'Pune': (18.5204, 73.8567, 95),
        'Lucknow': (26.8467, 80.9462, 180),
        'Kanpur': (26.4499, 80.3319, 190)
    }
    
    all_data = []
    
    for city_name, (lat, lon, base_aqi) in cities.items():
        print(f'Generating data for {city_name}...')
        df = generate_aqi_data(city_name, lat, lon, days=730, base_aqi=base_aqi)
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    output_path = 'data/raw/aqi_training_data.csv'
    combined_df.to_csv(output_path, index=False)
    
    print(f'\nData generated successfully!')
    print(f'Total records: {len(combined_df)}')
    print(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    print(f"Cities: {combined_df['city'].unique()}")
    print(f'\nSaved to: {output_path}')

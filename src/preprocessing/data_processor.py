import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

class AQIPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_columns = None
        self.target_column = 'pm2_5'
        
    def load_data(self, filepath):
        '''Load CSV data'''
        print(f'Loading data from {filepath}...')
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def create_time_features(self, df):
        '''Extract time-based features'''
        df = df.copy()
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for hour and month
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def handle_missing_values(self, df):
        '''Handle missing values'''
        print(f'Missing values before: {df.isnull().sum().sum()}')
        
        # Forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        print(f'Missing values after: {df.isnull().sum().sum()}')
        return df
    
    def create_lag_features(self, df, target_col='pm2_5', lags=[1, 2, 3, 6, 12, 24]):
        '''Create lag features for time series'''
        df = df.copy()
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df.groupby('city')[target_col].shift(lag)
        
        # Rolling statistics
        for window in [6, 12, 24]:
            df[f'{target_col}_rolling_mean_{window}'] = df.groupby('city')[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'{target_col}_rolling_std_{window}'] = df.groupby('city')[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        
        return df
    
    def prepare_sequences(self, df, seq_length=24, forecast_horizon=6):
        '''Prepare sequences for LSTM'''
        # Sort by city and timestamp
        df = df.sort_values(['city', 'timestamp'])
        
        # Select feature columns
        feature_cols = [
            'pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3', 'no', 'nh3',
            'temperature', 'humidity', 'wind_speed', 'pressure',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'is_weekend'
        ]
        
        # Add lag features if they exist
        lag_cols = [col for col in df.columns if 'lag' in col or 'rolling' in col]
        feature_cols.extend(lag_cols)
        
        # Remove duplicates
        feature_cols = list(dict.fromkeys(feature_cols))
        
        # Filter only existing columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        self.feature_columns = feature_cols
        
        X_sequences = []
        y_sequences = []
        
        for city in df['city'].unique():
            city_data = df[df['city'] == city][feature_cols].values
            target_data = df[df['city'] == city]['pm2_5'].values
            
            for i in range(len(city_data) - seq_length - forecast_horizon + 1):
                X_sequences.append(city_data[i:i + seq_length])
                y_sequences.append(target_data[i + seq_length:i + seq_length + forecast_horizon])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def fit_transform(self, X):
        '''Fit scaler and transform data'''
        # Reshape for scaling
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        
        # Fit and transform
        X_scaled = self.scaler.fit_transform(X_reshaped)
        
        # Reshape back
        return X_scaled.reshape(n_samples, n_timesteps, n_features)
    
    def transform(self, X):
        '''Transform data using fitted scaler'''
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_reshaped)
        return X_scaled.reshape(n_samples, n_timesteps, n_features)
    
    def save_scaler(self, filepath='models/scaler.pkl'):
        '''Save the fitted scaler'''
        joblib.dump(self.scaler, filepath)
        print(f'Scaler saved to {filepath}')
    
    def load_scaler(self, filepath='models/scaler.pkl'):
        '''Load a fitted scaler'''
        self.scaler = joblib.load(filepath)
        print(f'Scaler loaded from {filepath}')

def preprocess_pipeline(input_file, output_dir='data/processed'):
    '''Complete preprocessing pipeline'''
    
    # Initialize preprocessor
    preprocessor = AQIPreprocessor()
    
    # Load data
    df = preprocessor.load_data(input_file)
    print(f'Loaded {len(df)} records')
    
    # Create time features
    df = preprocessor.create_time_features(df)
    print('Created time features')
    
    # Create lag features
    df = preprocessor.create_lag_features(df)
    print('Created lag features')
    
    # Handle missing values
    df = preprocessor.handle_missing_values(df)
    
    # Drop rows with NaN (from lag features)
    df = df.dropna()
    print(f'Records after dropping NaN: {len(df)}')
    
    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    processed_file = os.path.join(output_dir, 'aqi_processed.csv')
    df.to_csv(processed_file, index=False)
    print(f'Processed data saved to {processed_file}')
    
    # Prepare sequences
    print('\nPreparing sequences for LSTM...')
    X, y = preprocessor.prepare_sequences(df, seq_length=24, forecast_horizon=6)
    print(f'X shape: {X.shape}')
    print(f'y shape: {y.shape}')
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f'\nTrain set: {X_train.shape[0]} samples')
    print(f'Validation set: {X_val.shape[0]} samples')
    print(f'Test set: {X_test.shape[0]} samples')
    
    # Scale the data
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(X_test)
    
    # Save scaler
    preprocessor.save_scaler('models/scaler.pkl')
    
    # Save processed arrays
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train_scaled)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val_scaled)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test_scaled)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    print(f'\nAll processed data saved to {output_dir}')
    
    return preprocessor, X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

if __name__ == '__main__':
    # Run preprocessing
    preprocessor, X_train, X_val, X_test, y_train, y_val, y_test = preprocess_pipeline(
        'data/raw/aqi_training_data.csv'
    )
    
    print('\nPreprocessing completed successfully!')

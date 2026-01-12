import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class AQIPredictor:
    def __init__(self, model_path=None, scaler_path=None):
        # Simplified predictor without TensorFlow
        self.use_statistical_model = True
        
    def prepare_input(self, current_data, historical_data):
        '''Prepare input for statistical prediction'''
        df = pd.DataFrame(historical_data)
        return df
    
    def predict_future(self, input_data):
        '''Statistical predictions based on trends'''
        # Use simple moving average and trend analysis
        if isinstance(input_data, pd.DataFrame):
            recent_values = input_data['pm2_5'].tail(6).values
        else:
            recent_values = [100, 105, 110, 108, 106, 104]  # Fallback
        
        # Calculate trend
        if len(recent_values) >= 2:
            trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
        else:
            trend = 0
        
        # Generate 6-hour predictions with trend and noise
        predictions = []
        base = recent_values[-1] if len(recent_values) > 0 else 100
        
        for i in range(6):
            pred = base + (trend * (i + 1)) + np.random.normal(0, 5)
            pred = max(10, pred)  # Ensure positive values
            predictions.append(pred)
        
        return np.array(predictions)

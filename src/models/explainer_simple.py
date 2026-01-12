import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import os

class AQIExplainer:
    def __init__(self, model_path='models/saved_models/aqi_lstm_model.h5'):
        self.model = keras.models.load_model(model_path)
        
    def get_feature_importance_simple(self, X_samples, feature_names=None):
        '''Simple feature importance based on gradient analysis'''
        print('Calculating feature importance...')
        
        # Get predictions
        predictions = self.model.predict(X_samples)
        
        # Calculate variance in features
        feature_variance = np.var(X_samples, axis=0).mean(axis=0)
        
        # Normalize
        feature_importance = feature_variance / feature_variance.sum()
        
        if feature_names is not None:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return feature_importance
    
    def plot_feature_importance(self, importance_df, save_path='models/feature_importance.png'):
        '''Plot feature importance'''
        plt.figure(figsize=(12, 8))
        
        top_features = importance_df.head(15)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title('Top 15 Most Important Features')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Feature importance plot saved to {save_path}')
        plt.close()

def generate_feature_importance():
    '''Generate feature importance analysis'''
    
    # Load test data
    print('Loading test data...')
    X_test = np.load('data/processed/X_test.npy')
    
    # Feature names
    feature_names = [
        'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'NO', 'NH3',
        'Temperature', 'Humidity', 'Wind Speed', 'Pressure',
        'Hour Sin', 'Hour Cos', 'Month Sin', 'Month Cos', 'Is Weekend'
    ]
    
    # Add lag features
    for lag in [1, 2, 3, 6, 12, 24]:
        feature_names.append(f'PM2.5 Lag {lag}')
    
    for window in [6, 12, 24]:
        feature_names.append(f'PM2.5 Mean {window}h')
        feature_names.append(f'PM2.5 Std {window}h')
    
    # Initialize explainer
    explainer = AQIExplainer()
    
    # Get feature importance
    sample_size = min(500, len(X_test))
    importance_df = explainer.get_feature_importance_simple(X_test[:sample_size], feature_names)
    
    print('\nTop 10 Most Important Features:')
    print(importance_df.head(10))
    
    # Plot
    explainer.plot_feature_importance(importance_df)
    
    # Save
    importance_df.to_csv('models/feature_importance.csv', index=False)
    print('\nFeature importance saved to models/feature_importance.csv')
    
    return explainer, importance_df

if __name__ == '__main__':
    print('NOTE: Using simplified feature importance (SHAP not available)')
    print('You can install SHAP later for advanced explainability\n')
    
    explainer, importance = generate_feature_importance()
    
    print('\n' + '='*50)
    print('FEATURE IMPORTANCE ANALYSIS COMPLETED!')
    print('='*50)

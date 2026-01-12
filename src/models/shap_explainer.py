import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from tensorflow import keras
import joblib
import os

class AQIExplainer:
    def __init__(self, model_path='models/saved_models/aqi_lstm_model.h5'):
        self.model = keras.models.load_model(model_path)
        self.explainer = None
        self.shap_values = None
        
    def create_explainer(self, background_data, max_samples=100):
        '''Create SHAP explainer'''
        print('Creating SHAP explainer (this may take a few minutes)...')
        
        # Use a subset for background
        if len(background_data) > max_samples:
            indices = np.random.choice(len(background_data), max_samples, replace=False)
            background = background_data[indices]
        else:
            background = background_data
        
        # Create explainer
        self.explainer = shap.DeepExplainer(self.model, background)
        print('SHAP explainer created successfully!')
        
    def explain_predictions(self, X_samples, feature_names=None):
        '''Generate SHAP values for samples'''
        print('Calculating SHAP values...')
        
        if self.explainer is None:
            raise ValueError('Explainer not created. Call create_explainer first.')
        
        # Calculate SHAP values
        self.shap_values = self.explainer.shap_values(X_samples)
        
        print('SHAP values calculated!')
        return self.shap_values
    
    def plot_summary(self, X_samples, feature_names=None, max_display=20, save_path='models/shap_summary.png'):
        '''Create SHAP summary plot'''
        if self.shap_values is None:
            self.explain_predictions(X_samples, feature_names)
        
        # Reshape for plotting (take last timestep)
        shap_values_2d = self.shap_values[0][:, -1, :]
        X_2d = X_samples[:, -1, :]
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values_2d,
            X_2d,
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'SHAP summary plot saved to {save_path}')
        plt.close()
    
    def plot_waterfall(self, X_sample, feature_names=None, save_path='models/shap_waterfall.png'):
        '''Create SHAP waterfall plot for a single prediction'''
        if self.shap_values is None:
            self.explain_predictions(X_sample.reshape(1, *X_sample.shape), feature_names)
        
        # Take the first output and last timestep
        shap_values_1d = self.shap_values[0][0, -1, :]
        X_1d = X_sample[-1, :]
        
        plt.figure(figsize=(10, 8))
        
        # Create explanation object
        explanation = shap.Explanation(
            values=shap_values_1d,
            base_values=self.explainer.expected_value[0],
            data=X_1d,
            feature_names=feature_names
        )
        
        shap.waterfall_plot(explanation, max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'SHAP waterfall plot saved to {save_path}')
        plt.close()
    
    def get_feature_importance(self, X_samples, feature_names=None):
        '''Get feature importance based on mean absolute SHAP values'''
        if self.shap_values is None:
            self.explain_predictions(X_samples, feature_names)
        
        # Calculate mean absolute SHAP values
        shap_values_2d = self.shap_values[0][:, -1, :]
        feature_importance = np.abs(shap_values_2d).mean(axis=0)
        
        if feature_names is not None:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return feature_importance

def generate_shap_explanations():
    '''Generate SHAP explanations for the model'''
    
    # Load test data
    print('Loading test data...')
    X_test = np.load('data/processed/X_test.npy')
    
    # Load feature names (you can customize this)
    feature_names = [
        'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'NO', 'NH3',
        'Temperature', 'Humidity', 'Wind Speed', 'Pressure',
        'Hour Sin', 'Hour Cos', 'Month Sin', 'Month Cos', 'Is Weekend'
    ]
    
    # Add lag feature names
    for lag in [1, 2, 3, 6, 12, 24]:
        feature_names.append(f'PM2.5 Lag {lag}')
    
    for window in [6, 12, 24]:
        feature_names.append(f'PM2.5 Mean {window}h')
        feature_names.append(f'PM2.5 Std {window}h')
    
    # Initialize explainer
    explainer = AQIExplainer()
    
    # Create explainer with background data
    explainer.create_explainer(X_test[:100])
    
    # Generate SHAP values for subset
    sample_size = min(200, len(X_test))
    X_samples = X_test[:sample_size]
    
    # Explain predictions
    explainer.explain_predictions(X_samples, feature_names)
    
    # Create visualizations
    print('\nGenerating SHAP visualizations...')
    
    # Summary plot
    explainer.plot_summary(X_samples, feature_names, save_path='models/shap_summary.png')
    
    # Waterfall plot for first sample
    explainer.plot_waterfall(X_samples[0], feature_names, save_path='models/shap_waterfall.png')
    
    # Get feature importance
    importance_df = explainer.get_feature_importance(X_samples, feature_names)
    print('\nTop 10 Most Important Features:')
    print(importance_df.head(10))
    
    # Save importance
    importance_df.to_csv('models/feature_importance.csv', index=False)
    print('\nFeature importance saved to models/feature_importance.csv')
    
    return explainer, importance_df

if __name__ == '__main__':
    explainer, importance = generate_shap_explanations()
    
    print('\n' + '='*50)
    print('SHAP ANALYSIS COMPLETED!')
    print('='*50)

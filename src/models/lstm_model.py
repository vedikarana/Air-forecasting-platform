import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
from datetime import datetime

class AQILSTMModel:
    def __init__(self, input_shape, forecast_horizon=6):
        self.input_shape = input_shape
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.history = None
        
    def build_model(self, lstm_units=[128, 64], dropout_rate=0.2):
        '''Build LSTM model architecture'''
        
        model = keras.Sequential([
            # First LSTM layer
            layers.LSTM(
                lstm_units[0],
                return_sequences=True,
                input_shape=self.input_shape
            ),
            layers.Dropout(dropout_rate),
            layers.BatchNormalization(),
            
            # Second LSTM layer
            layers.LSTM(lstm_units[1], return_sequences=False),
            layers.Dropout(dropout_rate),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(dropout_rate),
            
            layers.Dense(32, activation='relu'),
            
            # Output layer
            layers.Dense(self.forecast_horizon)
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        '''Compile the model'''
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        print(self.model.summary())
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        '''Train the model'''
        
        # Create callbacks
        checkpoint_dir = 'models/checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train the model
        print('\nStarting training...')
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print('\nTraining completed!')
        return self.history
    
    def evaluate(self, X_test, y_test):
        '''Evaluate the model'''
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        print('\nTest Results:')
        print(f'Loss: {results[0]:.4f}')
        print(f'MAE: {results[1]:.4f}')
        print(f'MSE: {results[2]:.4f}')
        print(f'RMSE: {np.sqrt(results[2]):.4f}')
        
        return results
    
    def predict(self, X):
        '''Make predictions'''
        return self.model.predict(X)
    
    def plot_training_history(self, save_path='models/training_history.png'):
        '''Plot training history'''
        if self.history is None:
            print('No training history available')
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Model Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # MAE plot
        axes[1].plot(self.history.history['mae'], label='Train MAE')
        axes[1].plot(self.history.history['val_mae'], label='Val MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Model MAE')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Training history plot saved to {save_path}')
        plt.close()
    
    def save_model(self, filepath='models/saved_models/aqi_lstm_model.h5'):
        '''Save the trained model'''
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f'Model saved to {filepath}')
    
    def load_model(self, filepath='models/saved_models/aqi_lstm_model.h5'):
        '''Load a trained model'''
        self.model = keras.models.load_model(filepath)
        print(f'Model loaded from {filepath}')

def train_aqi_model():
    '''Complete training pipeline'''
    
    # Load processed data
    print('Loading processed data...')
    X_train = np.load('data/processed/X_train.npy')
    X_val = np.load('data/processed/X_val.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_val = np.load('data/processed/y_val.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    print(f'X_train shape: {X_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    
    # Initialize model
    input_shape = (X_train.shape[1], X_train.shape[2])
    forecast_horizon = y_train.shape[1]
    
    model = AQILSTMModel(input_shape=input_shape, forecast_horizon=forecast_horizon)
    
    # Build and compile
    model.build_model(lstm_units=[128, 64], dropout_rate=0.3)
    model.compile_model(learning_rate=0.001)
    
    # Train
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=64
    )
    
    # Evaluate
    results = model.evaluate(X_test, y_test)
    
    # Plot training history
    model.plot_training_history()
    
    # Save model
    model.save_model()
    
    # Make sample predictions
    print('\nMaking sample predictions...')
    y_pred = model.predict(X_test[:5])
    
    print('\nSample predictions vs actual:')
    for i in range(5):
        print(f'\nSample {i+1}:')
        print(f'Predicted: {y_pred[i]}')
        print(f'Actual: {y_test[i]}')
    
    return model, results

if __name__ == '__main__':
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Train the model
    model, results = train_aqi_model()
    
    print('\n' + '='*50)
    print('MODEL TRAINING COMPLETED SUCCESSFULLY!')
    print('='*50)

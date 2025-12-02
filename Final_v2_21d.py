
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime
import warnings
import json
import os
import pickle
warnings.filterwarnings('ignore')

# Deep Learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input,
                                      Bidirectional, Concatenate,
                                      BatchNormalization, GRU)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# Scikit-learn imports
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (15, 8)


class StockDataLoader:
    """Advanced stock data loader with technical indicators"""

    def __init__(self, ticker='AAPL', start_date='2015-01-01', end_date=None):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None

    def download_data(self):
        """Download stock data from Yahoo Finance"""
        print(f"Downloading {self.ticker} data from {self.start_date} to {self.end_date}...")

        try:
            ticker = yf.Ticker(self.ticker)
            self.data = ticker.history(start=self.start_date, end=self.end_date)

            if len(self.data) == 0:
                self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date,
                                       progress=False, threads=False)

            if len(self.data) == 0:
                raise ValueError("No data downloaded")

            print(f"Downloaded {len(self.data)} trading days")
            return self.data

        except Exception as e:
            print(f"Error downloading from Yahoo Finance: {e}")
            raise

    def add_technical_indicators(self):
        """Add comprehensive technical indicators"""
        df = self.data.copy()

        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # RSI
        for period in [14, 21]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        for period in [20]:
            df[f'BB_Middle_{period}'] = df['Close'].rolling(window=period).mean()
            bb_std = df['Close'].rolling(window=period).std()
            df[f'BB_Upper_{period}'] = df[f'BB_Middle_{period}'] + (bb_std * 2)
            df[f'BB_Lower_{period}'] = df[f'BB_Middle_{period}'] - (bb_std * 2)
            df[f'BB_Width_{period}'] = df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']
            df[f'BB_Position_{period}'] = (df['Close'] - df[f'BB_Lower_{period}']) / df[f'BB_Width_{period}']

        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stochastic_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()

        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR_14'] = true_range.rolling(14).mean()

        # OBV
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

        # Volume indicators
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']

        # Price momentum
        for period in [1, 5, 10, 20]:
            df[f'Momentum_{period}'] = df['Close'].diff(period)
            df[f'ROC_{period}'] = df['Close'].pct_change(period) * 100

        # Volatility
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        df['Volatility_50'] = df['Returns'].rolling(window=50).std()

        # Price channels
        df['High_20'] = df['High'].rolling(window=20).max()
        df['Low_20'] = df['Low'].rolling(window=20).min()
        df['Channel_Position'] = (df['Close'] - df['Low_20']) / (df['High_20'] - df['Low_20'])

        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)

        # Time-based features
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['Day_of_Month'] = df.index.day

        # Drop NaN values
        df = df.dropna()

        self.data = df
        print(f"Added technical indicators. Total features: {len(df.columns)}")
        return df


class AdvancedStockPredictor:
    """Hybrid LSTM-GRU stock predictor for 21-day prediction"""

    def __init__(self, sequence_length=60, prediction_days=21, features_to_use='all'):
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        self.features_to_use = features_to_use
        self.scalers = {}
        self.feature_names = []
        self.models = []  # Ensemble of models (as dictionaries!)

    def prepare_data(self, df, target_column='Close'):
        """Prepare data with proper train/test split to prevent data leakage"""

        # Select features
        if self.features_to_use == 'all':
            exclude_cols = ['Adj Close']
            self.feature_names = [col for col in df.columns if col not in exclude_cols]
        else:
            self.feature_names = self.features_to_use

        data = df[self.feature_names].values

        # Split data chronologically
        split_index = int(len(data) * 0.8)
        train_data = data[:split_index]
        test_data = data[split_index:]

        # Fit scalers only on training data
        self.scalers['features'] = RobustScaler()
        train_data_scaled = self.scalers['features'].fit_transform(train_data)
        test_data_scaled = self.scalers['features'].transform(test_data)

        # Find target column index
        self.target_idx = self.feature_names.index(target_column)

        # Create sequences
        X_train, y_train = self.create_sequences(train_data_scaled)
        X_test, y_test = self.create_sequences(test_data_scaled)

        print(f"\nData preparation complete:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Sequence length: {self.sequence_length} days")
        print(f"  Features per timestep: {X_train.shape[2]}")
        print(f"  Prediction horizon: {self.prediction_days} days")

        return X_train, y_train, X_test, y_test, split_index

    def create_sequences(self, data):
        """Create sequences for 21-day prediction"""
        X, y = [], []

        for i in range(self.sequence_length, len(data) - self.prediction_days + 1):
            X.append(data[i - self.sequence_length:i])
            #output: next 21 days closing prices
            y.append(data[i:i + self.prediction_days, self.target_idx])

        return np.array(X), np.array(y)

    def build_hybrid_model(self, input_shape, lstm_units=[128, 96], dropout=0.35, learning_rate=0.0008, seed=42):
        """
        Build Hybrid LSTM-GRU model for 21-day prediction
        Winner configuration: [128, 96] units with dropout 0.35
        """
        np.random.seed(seed)
        tf.random.set_seed(seed)

        inputs = Input(shape=input_shape)

        # LSTM branch
        lstm_out = LSTM(lstm_units[0], return_sequences=True)(inputs)
        lstm_out = Dropout(dropout)(lstm_out)
        if len(lstm_units) > 1:
            lstm_out = LSTM(lstm_units[1], return_sequences=False)(lstm_out)
            lstm_out = Dropout(dropout)(lstm_out)
        else:
            lstm_out = LSTM(lstm_units[0] // 2, return_sequences=False)(lstm_out)
            lstm_out = Dropout(dropout)(lstm_out)

        # GRU branch
        gru_out = GRU(lstm_units[0], return_sequences=True)(inputs)
        gru_out = Dropout(dropout)(gru_out)
        if len(lstm_units) > 1:
            gru_out = GRU(lstm_units[1], return_sequences=False)(gru_out)
            gru_out = Dropout(dropout)(gru_out)
        else:
            gru_out = GRU(lstm_units[0] // 2, return_sequences=False)(gru_out)
            gru_out = Dropout(dropout)(gru_out)

        # Merge branches
        merged = Concatenate()([lstm_out, gru_out])
        merged = BatchNormalization()(merged)
        merged = Dense(64, activation='relu')(merged)
        merged = Dropout(dropout)(merged)

        # Output layer for 21-day prediction
        outputs = Dense(self.prediction_days)(merged)

        model = Model(inputs=inputs, outputs=outputs)

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        return model

    def train_single_model(self, X_train, y_train, X_val, y_val, epochs=150, batch_size=32,
                          lstm_units=[128, 96], dropout=0.35, learning_rate=0.0008, seed=42):
        """Train a single model"""

        input_shape = (X_train.shape[1], X_train.shape[2])
        model = self.build_hybrid_model(input_shape, lstm_units, dropout, learning_rate, seed)

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=0
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=0
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        # Calculate validation MAE
        val_predictions = model.predict(X_val, verbose=0)
        val_mae = mean_absolute_error(y_val.flatten(), val_predictions.flatten())

        return model, val_mae, history

    def train_ensemble(self, X_train, y_train, X_test, y_test, epochs=150, batch_size=32):
        """
        Train ensemble of 7 models and select best 3
        """
        print("\n" + "="*70)
        print("TRAINING ENSEMBLE OF 7 MODELS")
        print("="*70)

        # Split validation from training
        val_split = int(len(X_train) * 0.9)
        X_val = X_train[val_split:]
        y_val = y_train[val_split:]
        X_train_subset = X_train[:val_split]
        y_train_subset = y_train[:val_split]

        all_models = []
        base_seeds = [100, 110, 120, 130, 140, 150, 160]

        for i, seed in enumerate(base_seeds, 1):
            print(f"\nTraining Model {i}/7 (seed={seed})...")

            model, val_mae, history = self.train_single_model(
                X_train_subset, y_train_subset,
                X_val, y_val,
                epochs=epochs,
                batch_size=batch_size,
                lstm_units=[128, 96],  # best config
                dropout=0.35,
                learning_rate=0.0008,
                seed=seed
            )

            all_models.append({
                'model': model,
                'history': history,
                'val_mae': val_mae,
                'seed': seed
            })

            print(f"  Model {i}: Val MAE = ${val_mae:.4f}")

            K.clear_session()

        # Sort by validation MAE and select best 3
        all_models.sort(key=lambda x: x['val_mae'])
        best_3_models = all_models[:3]

        self.models = best_3_models

        print("\n" + "="*70)
        print("FINAL ENSEMBLE SELECTED")
        print("="*70)
        for i, model_info in enumerate(self.models, 1):
            print(f"  {i}. HYBRID_{i}: Val MAE = ${model_info['val_mae']:.4f} (seed {model_info['seed']})")

        print("="*70)
        print("All 7 models ranked by validation MAE:")
        for i, model_info in enumerate(all_models, 1):
            marker = "✓" if i <= 3 else " "
            print(f"  {marker} Rank {i}: HYBRID_{i} - Val MAE ${model_info['val_mae']:.4f}")

        return all_models

    def predict_ensemble(self, X):
        """Make predictions using ensemble averaging"""
        predictions = []

        for model_info in self.models:
            pred = model_info['model'].predict(X, verbose=0)
            predictions.append(pred)

        # Average predictions from all models
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred

    def walk_forward_validation(self, X_test, y_test, n_steps=50):
        """
        Walk-forward validation to prevent data leakage
        """
        print("\n" + "="*70)
        print("WALK-FORWARD VALIDATION")
        print("="*70)

        predictions = []
        actuals = []

        # Use only the first n_steps for testing
        n_test = min(n_steps, len(X_test))

        for i in range(n_test):
            # Use only data up to current point
            X_current = X_test[i:i+1]
            y_current = y_test[i]

            # Make prediction using ensemble
            y_pred = self.predict_ensemble(X_current)

            predictions.append(y_pred[0])
            actuals.append(y_current)

            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{n_test} predictions completed")

        return np.array(predictions), np.array(actuals)

    def inverse_transform_predictions(self, predictions):
        """Convert scaled predictions back to original price scale"""
        # Create dummy array with all features
        dummy = np.zeros((len(predictions), len(self.feature_names)))

        # For each prediction day
        if len(predictions.shape) == 2:  # Multi-day predictions
            result = []
            for day_idx in range(predictions.shape[1]):
                dummy[:, self.target_idx] = predictions[:, day_idx]
                inversed = self.scalers['features'].inverse_transform(dummy)
                result.append(inversed[:, self.target_idx])
            return np.array(result).T
        else:  # Single day predictions
            dummy[:, self.target_idx] = predictions
            inversed = self.scalers['features'].inverse_transform(dummy)
            return inversed[:, self.target_idx]

    def save_models_and_config(self, models_dir, ticker, metrics):
        """
        Save all 3 best models and configuration
        """
        print("\n" + "="*70)
        print("SAVING MODELS AND CONFIGURATION")
        print("="*70)

        # Create directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)

        # Save each of the 3 best models
        for i, model_info in enumerate(self.models, 1):
            model_path = os.path.join(models_dir, f'model_{i}.keras')
            model_info['model'].save(model_path)
            print(f"Saved Model {i} to: {model_path}")

        # Save scalers
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scalers['features'], f)
        print(f"Saved scaler to: {scaler_path}")

        # Create configuration dictionary
        config = {
            'ticker': ticker,
            'sequence_length': self.sequence_length,
            'prediction_days': self.prediction_days,
            'feature_names': self.feature_names,
            'target_column': 'Close',
            'target_idx': self.target_idx,
            'architecture': {
                'type': 'Hybrid LSTM-GRU',
                'lstm_units': [128, 96],
                'dropout': 0.35,
                'learning_rate': 0.0008
            },
            'ensemble_info': {
                'total_models_trained': 7,
                'models_selected': 3,
                'selection_criteria': 'Lowest validation MAE',
                'model_seeds': [m['seed'] for m in self.models],
                'model_val_maes': [float(m['val_mae']) for m in self.models]
            },
            'performance_metrics': {
                'overall_mae': float(metrics['MAE']),
                'overall_rmse': float(metrics['RMSE']),
                'overall_mape': float(metrics['MAPE']),
                'overall_r2': float(metrics['R2']),
                'directional_accuracy': float(metrics['Directional_Accuracy'])
            },
            'per_day_metrics': {
                f'day_{day}': {
                    'mae': float(day_metrics['MAE']),
                    'rmse': float(day_metrics['RMSE']),
                    'mape': float(day_metrics['MAPE']),
                    'r2': float(day_metrics['R2'])
                }
                for day, day_metrics in metrics['Per_Day_Metrics'].items()
            }
        }

        # Save configuration as JSON
        config_path = os.path.join(models_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"✓ Saved configuration to: {config_path}")

        print("\nAll models and configuration saved successfully!")
        print("="*70)


def calculate_comprehensive_metrics(y_true, y_pred, prediction_day=None):
    """Calculate comprehensive evaluation metrics"""

    # Flatten for overall metrics
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    mape = mean_absolute_percentage_error(y_true_flat, y_pred_flat) * 100
    r2 = r2_score(y_true_flat, y_pred_flat)
    mse = mean_squared_error(y_true_flat, y_pred_flat)

    # Directional Accuracy
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_direction = np.sign(y_true[:, -1] - y_true[:, 0])
        y_pred_direction = np.sign(y_pred[:, -1] - y_pred[:, 0])
    else:
        y_true_direction = np.sign(np.diff(y_true_flat))
        y_pred_direction = np.sign(np.diff(y_pred_flat))

    directional_accuracy = np.mean(y_true_direction == y_pred_direction) * 100

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MSE': mse,
        'MAPE': mape,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy
    }

    #calculate metrics for each day
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        day_metrics = {}
        for day in range(y_true.shape[1]):
            day_mae = mean_absolute_error(y_true[:, day], y_pred[:, day])
            day_mape = mean_absolute_percentage_error(y_true[:, day], y_pred[:, day]) * 100
            day_r2 = r2_score(y_true[:, day], y_pred[:, day])
            day_rmse = np.sqrt(mean_squared_error(y_true[:, day], y_pred[:, day]))
            day_metrics[f'Day_{day+1}'] = {
                'MAE': day_mae,
                'MAPE': day_mape,
                'R2': day_r2,
                'RMSE': day_rmse
            }
        metrics['Per_Day_Metrics'] = day_metrics

    return metrics


def generate_thesis_graphs(y_true, y_pred, ticker, graphs_dir, prediction_days=21):
    """
    Generate graphs
    """
    print("\n" + "="*70)
    print("GENERATING THESIS GRAPHS")
    print("="*70)

    # Create directory
    os.makedirs(graphs_dir, exist_ok=True)

    # Predicted vs Actual Prices
    print("\nGenerating Graph 1: Predicted vs Actual Prices...")

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Plot Day 1 predictions
    ax1 = axes[0]
    ax1.plot(y_true[:, 0], label='Actual Price (Day 1)', color='blue', linewidth=2, alpha=0.7)
    ax1.plot(y_pred[:, 0], label='Predicted Price (Day 1)', color='red', linewidth=2, linestyle='--', alpha=0.7)
    ax1.set_title(f'{ticker} - Day 1 Prediction: Actual vs Predicted Prices', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot Day 21 predictions
    ax2 = axes[1]
    ax2.plot(y_true[:, -1], label=f'Actual Price (Day {prediction_days})', color='blue', linewidth=2, alpha=0.7)
    ax2.plot(y_pred[:, -1], label=f'Predicted Price (Day {prediction_days})', color='red', linewidth=2, linestyle='--', alpha=0.7)
    ax2.set_title(f'{ticker} - Day {prediction_days} Prediction: Actual vs Predicted Prices', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sample Index', fontsize=12)
    ax2.set_ylabel('Price ($)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    graph1_path = os.path.join(graphs_dir, 'predicted_vs_actual.png')
    plt.savefig(graph1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to: {graph1_path}")

    #Prediction Errors / Residuals
    print("\nGenerating Graph 2: Prediction Errors (Residuals)...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Calculate residuals for Day 1 and Day 21
    residuals_day1 = y_true[:, 0] - y_pred[:, 0]
    residuals_day21 = y_true[:, -1] - y_pred[:, -1]

    # Day 1 Residuals over time
    ax1 = axes[0, 0]
    ax1.plot(residuals_day1, color='purple', linewidth=1.5, alpha=0.7)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax1.set_title('Day 1 Prediction Errors Over Time', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Sample Index', fontsize=11)
    ax1.set_ylabel('Error ($)', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Day 1 Residuals distribution
    ax2 = axes[0, 1]
    ax2.hist(residuals_day1, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax2.set_title('Day 1 Error Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Error ($)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Day 21 Residuals over time
    ax3 = axes[1, 0]
    ax3.plot(residuals_day21, color='orange', linewidth=1.5, alpha=0.7)
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax3.set_title(f'Day {prediction_days} Prediction Errors Over Time', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Sample Index', fontsize=11)
    ax3.set_ylabel('Error ($)', fontsize=11)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Day 21 Residuals distribution
    ax4 = axes[1, 1]
    ax4.hist(residuals_day21, bins=30, color='orange', alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax4.set_title(f'Day {prediction_days} Error Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Error ($)', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    graph2_path = os.path.join(graphs_dir, 'prediction_errors.png')
    plt.savefig(graph2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved to: {graph2_path}")

    # Cumulative Returns Comparison
    print("\nGenerating Graph 3: Cumulative Returns Comparison...")

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Calculate cumulative returns for Day 1
    actual_returns_day1 = np.cumsum((y_true[1:, 0] - y_true[:-1, 0]) / y_true[:-1, 0] * 100)
    pred_returns_day1 = np.cumsum((y_pred[1:, 0] - y_pred[:-1, 0]) / y_pred[:-1, 0] * 100)

    ax1 = axes[0]
    ax1.plot(actual_returns_day1, label='Actual Cumulative Returns', color='green', linewidth=2, alpha=0.7)
    ax1.plot(pred_returns_day1, label='Predicted Cumulative Returns', color='red', linewidth=2, linestyle='--', alpha=0.7)
    ax1.set_title(f'{ticker} - Day 1 Cumulative Returns (%)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    # Calculate cumulative returns for Day 21
    actual_returns_day21 = np.cumsum((y_true[1:, -1] - y_true[:-1, -1]) / y_true[:-1, -1] * 100)
    pred_returns_day21 = np.cumsum((y_pred[1:, -1] - y_pred[:-1, -1]) / y_pred[:-1, -1] * 100)

    ax2 = axes[1]
    ax2.plot(actual_returns_day21, label='Actual Cumulative Returns', color='green', linewidth=2, alpha=0.7)
    ax2.plot(pred_returns_day21, label='Predicted Cumulative Returns', color='red', linewidth=2, linestyle='--', alpha=0.7)
    ax2.set_title(f'{ticker} - Day {prediction_days} Cumulative Returns (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sample Index', fontsize=12)
    ax2.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    plt.tight_layout()
    graph3_path = os.path.join(graphs_dir, 'cumulative_returns.png')
    plt.savefig(graph3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to: {graph3_path}")

    print("\nAll thesis graphs generated successfully!")
    print("="*70)


def print_metrics_report(metrics, ticker):
    """Print comprehensive metrics report"""

    print("\n" + "="*70)
    print(f"COMPREHENSIVE EVALUATION METRICS FOR {ticker}")
    print("="*70)

    print("\nOVERALL PERFORMANCE (All 21 Days Combined):")
    print(f"  • MAE (Mean Absolute Error):        ${metrics['MAE']:.4f}")
    print(f"  • RMSE (Root Mean Squared Error):   ${metrics['RMSE']:.4f}")
    print(f"  • MSE (Mean Squared Error):          ${metrics['MSE']:.4f}")
    print(f"  • MAPE (Mean Absolute % Error):      {metrics['MAPE']:.4f}%")
    print(f"  • R² Score:                          {metrics['R2']:.4f}")
    print(f"  • Directional Accuracy:              {metrics['Directional_Accuracy']:.2f}%")

    if 'Per_Day_Metrics' in metrics:
        print("\nPER-DAY PREDICTION METRICS (Selected Days):")
        key_days = ['Day_1', 'Day_5', 'Day_10', 'Day_15', 'Day_21']
        for day in key_days:
            if day in metrics['Per_Day_Metrics']:
                day_metrics = metrics['Per_Day_Metrics'][day]
                print(f"\n  {day} Ahead:")
                print(f"    ├─ MAE:  ${day_metrics['MAE']:.4f}")
                print(f"    ├─ RMSE: ${day_metrics['RMSE']:.4f}")
                print(f"    ├─ MAPE: {day_metrics['MAPE']:.4f}%")
                print(f"    └─ R²:   {day_metrics['R2']:.4f}")

    # Highlight Day 21
    if 'Per_Day_Metrics' in metrics:
        day21 = metrics['Per_Day_Metrics']['Day_21']
        print("\n" + "="*70)
        print("DAY 21 PERFORMANCE (For Engineering Paper)")
        print("="*70)
        print(f"  • Day 21 MAE:   ${day21['MAE']:.4f}")
        print(f"  • Day 21 RMSE:  ${day21['RMSE']:.4f}")
        print(f"  • Day 21 MAPE:  {day21['MAPE']:.4f}%")
        print(f"  • Day 21 R²:    {day21['R2']:.4f}")

    print(f"\n21-Day Model Results:")
    print(f"  • Overall MAE:     ${metrics['MAE']:.4f}")
    print(f"  • Day 21 MAE:      ${day21['MAE']:.4f}")
    print(f"  • Overall MAPE:    {metrics['MAPE']:.4f}%")

    print("\n" + "="*70)

def main():
    """Main execution function"""

    print("\n" + "="*70)
    print("FINAL 21-DAY STOCK PREDICTION MODEL - ENHANCED FOR THESIS")
    print("Configuration: HYBRID_Medium_60d ([128, 96] units)")
    print("Ensemble: Best 3 out of 7 models")
    print("Winner from comprehensive testing")
    print("="*70 + "\n")

    # Configuration - WINNING PARAMETERS FROM TESTING
    TICKER = 'AAPL'
    START_DATE = '2015-01-01'
    SEQUENCE_LENGTH = 60
    PREDICTION_DAYS = 21      # 21-day prediction
    EPOCHS = 150
    BATCH_SIZE = 32

    # Paths for saving
    MODELS_DIR = r'C:\Users\cinek\PycharmProjects\LSTMS\models\21d'
    GRAPHS_DIR = r'C:\Users\cinek\PycharmProjects\LSTMS\graphs\21d'

    # Load and prepare data
    print("Step 1: Loading and preparing data...")
    loader = StockDataLoader(ticker=TICKER, start_date=START_DATE)
    df = loader.download_data()
    df = loader.add_technical_indicators()

    # Initialize predictor
    print("\nStep 2: Initializing predictor...")
    predictor = AdvancedStockPredictor(
        sequence_length=SEQUENCE_LENGTH,
        prediction_days=PREDICTION_DAYS,
        features_to_use='all'
    )

    # Prepare data
    print("\nStep 3: Preparing sequences...")
    X_train, y_train, X_test, y_test, split_index = predictor.prepare_data(df)

    # Train ensemble
    print("\nStep 4: Training ensemble")
    models = predictor.train_ensemble(
        X_train, y_train, X_test, y_test,
        epochs=EPOCHS, batch_size=BATCH_SIZE
    )

    # Make predictions using walk-forward validation
    print("\nStep 5: Making predictions with walk-forward validation...")
    y_pred_scaled, y_test_subset = predictor.walk_forward_validation(X_test, y_test, n_steps=50)

    # Inverse transform predictions to actual prices
    print("\nStep 6: Converting predictions to original price scale...")
    y_pred = predictor.inverse_transform_predictions(y_pred_scaled)
    y_true = predictor.inverse_transform_predictions(y_test_subset)

    #mCalculate metrics on actual prices
    print("\nStep 7: Calculating comprehensive metrics on actual prices...")
    metrics = calculate_comprehensive_metrics(y_true, y_pred)

    # Generate reports
    print("\nStep 8: Generating final report...")
    print_metrics_report(metrics, TICKER)

    # Save Model
    print("\nStep 9: Saving models and configuration...")
    predictor.save_models_and_config(MODELS_DIR, TICKER, metrics)

    # Genarage gaphs
    print("\nStep 10: Generating graphs...")
    generate_thesis_graphs(y_true, y_pred, TICKER, GRAPHS_DIR, PREDICTION_DAYS)

    print("\n" + "="*70)
    print("FINAL 21-DAY MODEL COMPLETE!")
    print("="*70)

    return predictor, metrics, models


if __name__ == "__main__":
    predictor, metrics, models = main()
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pickle
import json
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')


class UniversalBiasCorrector:
    """
    Bias correction for any prediction horizon (1d, 5d, 10d, 21d)
    """

    def __init__(self, models_dir, prediction_days):
        """
        Args:
            models_dir: Path to model directory
            prediction_days: Number of days model predict
        """
        self.models_dir = models_dir
        self.prediction_days = prediction_days
        self.models = []
        self.scaler = None
        self.config = None
        self.feature_names = []
        self.bias_correction_factor = None

        print(f"\n{'='*70}")
        print(f"INITIALIZING {prediction_days}-DAY PREDICTOR WITH BIAS CORRECTION")
        print(f"{'='*70}")

        # Load everything
        self._load_config()
        self._load_scaler()
        self._load_models()

        # Calculate bias
        self.bias_correction_factor = self._calculate_bias()

    def _load_config(self):
        """Load configuration"""
        config_path = os.path.join(self.models_dir, 'config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.feature_names = self.config['feature_names']
        self.sequence_length = self.config['sequence_length']
        self.target_idx = self.config['target_idx']

        print(f" Loaded config: {self.prediction_days}-day predictions")

    def _load_scaler(self):
        """Load scaler"""
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        print(f" Loaded scaler:")
        print(f"  Center: ${self.scaler.center_[0]:.2f}")
        print(f"  Scale: ${self.scaler.scale_[0]:.2f}")

    def _load_models(self):
        """Load ensemble models"""
        for i in range(1, 4):
            model_path = os.path.join(self.models_dir, f'model_{i}.keras')
            if os.path.exists(model_path):
                model = load_model(model_path)
                self.models.append(model)

        print(f" Loaded {len(self.models)} models")

        # Check if model output matches
        if len(self.models) > 0:
            # Make a test prediction to shape
            test_input = np.random.randn(1, self.sequence_length, len(self.feature_names))
            test_output = self.models[0].predict(test_input, verbose=0)
            actual_prediction_days = test_output.shape[1] if len(test_output.shape) > 1 else 1

            if actual_prediction_days != self.prediction_days:
                print(f"\n  MODEL MISMATCH DETECTED!")
                print(f"   Expected: {self.prediction_days}-day predictions")
                print(f"   Actual model outputs: {actual_prediction_days} days")
                print(f"   This model was trained for {actual_prediction_days}-day horizon")
                print(f"   Will use first {self.prediction_days} day(s) from predictions")

                # Update prediction_days to match model
                self.actual_model_days = actual_prediction_days
            else:
                self.actual_model_days = self.prediction_days
                print(f"Model outputs match expected {self.prediction_days}-day horizon")

    def _calculate_bias(self):
        """
        Calculate bias correction factor
        """
        print(f"\n{'='*70}")
        print(f"CALCULATING BIAS FOR {self.prediction_days}-DAY MODEL")
        print(f"{'='*70}")

        try:
            # Get current price
            ticker = yf.Ticker('AAPL')
            df = ticker.history(period='5d')
            current_price = df['Close'].iloc[-1]

            print(f"\nCurrent market price: ${current_price:.2f}")

            # Make a raw predictio
            raw_pred = self._make_raw_prediction('AAPL')

            if raw_pred is not None and len(raw_pred) > 0:
                # Use first day prediction for bias calculation
                typical_prediction = raw_pred[0]

                print(f"Raw model prediction (Day 1): ${typical_prediction:.2f}")

                # Calculate correction factor
                bias = current_price - typical_prediction
                correction_factor = current_price / typical_prediction

                print(f"\nSystematic bias: ${bias:.2f}")
                print(f"Correction factor: {correction_factor:.4f}")

                # Validate correction factor
                if correction_factor < 0.5 or correction_factor > 2.0:
                    print(f"\n WARNING: Correction factor {correction_factor:.4f} seems extreme!")
                    print(f"   Using conservative factor of 1.0 (no correction)")
                    correction_factor = 1.0
                else:
                    print(f"\nCorrection factor looks reasonable!")

                return correction_factor
            else:
                print(f"\n  Could not calculate bias, using factor 1.0")
                return 1.0

        except Exception as e:
            print(f"\n Error calculating bias: {e}")
            print(f"   Using correction factor 1.0 (no correction)")
            return 1.0

    def _make_raw_prediction(self, ticker):
        """Make prediction without bias correction"""
        try:
            # Download data
            DEMO_DATE = datetime(2025, 11, 18)
            end_date = DEMO_DATE
            start_date = end_date - timedelta(days=600)

            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(start=start_date, end=end_date)

            # Add indicators
            df = self._add_technical_indicators(df)

            if len(df) < self.sequence_length:
                return None

            # Prepare input
            df_features = df[self.feature_names]
            recent_data = df_features.values[-self.sequence_length:]

            # Scale
            scaled_data = self.scaler.transform(recent_data)
            X = scaled_data.reshape(1, self.sequence_length, len(self.feature_names))

            # Predict
            predictions_scaled = []
            for model in self.models:
                pred = model.predict(X, verbose=0)
                predictions_scaled.append(pred[0])

            ensemble_pred_scaled = np.mean(predictions_scaled, axis=0)

            # Check if model outputs match expected days
            actual_output_days = len(ensemble_pred_scaled)

            if actual_output_days != self.prediction_days:
                print(f"\n  WARNING: Model outputs {actual_output_days} days but expecting {self.prediction_days} days")
                print(f"   This model was trained for {actual_output_days}-day predictions")

                if actual_output_days > self.prediction_days:
                    # Model predicts MORE days than expected
                    print(f"   Using only first {self.prediction_days} day(s) from predictions")
                    ensemble_pred_scaled = ensemble_pred_scaled[:self.prediction_days]
                else:
                    # Model predicts fewer days than expected
                    print(f"   Padding predictions to {self.prediction_days} days")
                    padding = np.repeat(ensemble_pred_scaled[-1], self.prediction_days - actual_output_days)
                    ensemble_pred_scaled = np.concatenate([ensemble_pred_scaled, padding])

            # Inverse transform
            dummy = np.zeros((self.prediction_days, len(self.feature_names)))
            dummy[:, self.target_idx] = ensemble_pred_scaled
            inversed = self.scaler.inverse_transform(dummy)
            predictions = inversed[:, self.target_idx]

            return predictions

        except Exception as e:
            print(f"Error in raw prediction: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _add_technical_indicators(self, df):
        """Add all technical indicators"""
        # Price features
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
        df['BB_Middle_20'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper_20'] = df['BB_Middle_20'] + (bb_std * 2)
        df['BB_Lower_20'] = df['BB_Middle_20'] - (bb_std * 2)
        df['BB_Width_20'] = df['BB_Upper_20'] - df['BB_Lower_20']
        df['BB_Position_20'] = (df['Close'] - df['BB_Lower_20']) / df['BB_Width_20']

        # Stochastic
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

        # Volume
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']

        # Momentum
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

        # Time features
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['Day_of_Month'] = df.index.day

        df = df.dropna()
        return df

    def predict(self, ticker='AAPL'):
        """
        Make bias-corrected prediction

        Returns:
            predictions: Corrected price predictions
            metadata: Additional info
        """
        print(f"\n{'='*70}")
        print(f"MAKING {self.prediction_days}-DAY PREDICTION (BIAS-CORRECTED)")
        print(f"{'='*70}")

        # Make raw prediction
        raw_predictions = self._make_raw_prediction(ticker)

        if raw_predictions is None:
            return None, None

        # Get current price
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period='5d')
        current_price = df['Close'].iloc[-1]

        # Apply bias
        corrected_predictions = raw_predictions * self.bias_correction_factor

        # Calculate changes
        changes = corrected_predictions - current_price
        pct_changes = (changes / current_price) * 100

        # Print results
        print(f"\nCurrent Price: ${current_price:.2f}")
        print(f"Bias Correction Factor: {self.bias_correction_factor:.4f}")

        print(f"\nRAW Predictions (uncorrected):")
        for i in range(min(3, len(raw_predictions))):
            print(f"  Day {i+1}: ${raw_predictions[i]:.2f}")

        print(f"\nCORRECTED Predictions:")
        for i, (pred, change, pct_change) in enumerate(zip(corrected_predictions, changes, pct_changes), 1):
            sign = '+' if change >= 0 else ''
            print(f"  Day {i}: ${pred:.2f} ({sign}{change:.2f}, {sign}{pct_change:.2f}%)")

        metadata = {
            'ticker': ticker,
            'prediction_days': self.prediction_days,
            'current_price': float(current_price),
            'bias_correction_factor': float(self.bias_correction_factor),
            'scaler_center': float(self.scaler.center_[0]),
            'scaler_scale': float(self.scaler.scale_[0]),
            'warning': 'Bias correction applied'
        }

        return corrected_predictions, metadata


def test_all_models():
    """Test bias correction on all models"""

    print("\n" + "="*70)
    print("TESTING ALL MODELS WITH BIAS CORRECTION")
    print("="*70)

    # Define allmodels
    models_config = [
        {'dir': r'C:\Users\cinek\PycharmProjects\LSTMS\models\1d', 'days': 1},
        {'dir': r'C:\Users\cinek\PycharmProjects\LSTMS\models\5d', 'days': 5},
        {'dir': r'C:\Users\cinek\PycharmProjects\LSTMS\models\10d', 'days': 10},
        {'dir': r'C:\Users\cinek\PycharmProjects\LSTMS\models\21d', 'days': 21},
    ]

    results = {}

    for config in models_config:
        models_dir = config['dir']
        pred_days = config['days']

        print(f"\n{'='*70}")
        print(f"TESTING {pred_days}-DAY MODEL")
        print(f"{'='*70}")

        if not os.path.exists(models_dir):
            print(f" Model directory not found: {models_dir}")
            print(f"   Skipping...")
            continue

        try:
            # Initialize predictor with bias correction
            predictor = UniversalBiasCorrector(models_dir, pred_days)

            # Make prediction
            predictions, metadata = predictor.predict('AAPL')

            if predictions is not None:
                results[f'{pred_days}d'] = {
                    'predictions': predictions.tolist(),
                    'metadata': metadata
                }
                print(f"\n{pred_days}-day model: Success!")
            else:
                print(f"\n{pred_days}-day model: Failed to predict")

        except Exception as e:
            print(f"\n{pred_days}-day model: Error - {e}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF ALL MODELS")
    print("="*70)

    for model_name, result in results.items():
        meta = result['metadata']
        preds = result['predictions']
        current = meta['current_price']

        # Calculate average % change
        avg_change = np.mean([(p - current) / current * 100 for p in preds])

        print(f"\n{model_name.upper()} Model:")
        print(f"  Bias Correction Factor: {meta['bias_correction_factor']:.4f}")
        print(f"  Scaler Center: ${meta['scaler_center']:.2f}")
        print(f"  Average % Change: {avg_change:+.2f}%")

        if abs(avg_change) < 5:
            print(f" Looks good!")
        else:
            print(f" Might need adjustment")

    return results


def save_bias_corrections(results, save_path='bias_corrections.json'):
    """Save bias correction factors for each model"""

    corrections = {}
    for model_name, result in results.items():
        corrections[model_name] = {
            'bias_correction_factor': result['metadata']['bias_correction_factor'],
            'scaler_center': result['metadata']['scaler_center'],
            'scaler_scale': result['metadata']['scaler_scale']
        }

    with open(save_path, 'w') as f:
        json.dump(corrections, f, indent=4)

    print(f"\nSaved bias corrections to: {save_path}")
    print(f"   You can load these in your web app!")


if __name__ == "__main__":
    # Test all models
    results = test_all_models()

    # Save correction factors
    if results:
        save_bias_corrections(results)

        print("\n" + "="*70)
        print("BIAS CORRECTION SETUP COMPLETE!")
        print("="*70)
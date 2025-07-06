import sys
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import timedelta

# Debug execution context
print(f"Script location: {__file__}")
print(f"Current working directory: {os.getcwd()}")

# Set BASE_DIR relative to script location
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
print(f"Calculated BASE_DIR: {BASE_DIR}")

# Add BASE_DIR and src to sys.path
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'src'))
print(f"Updated sys.path: {sys.path}")

DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

print(f"DATA_DIR: {DATA_DIR}")
print(f"MODEL_DIR: {MODEL_DIR}")
print(
    f"Contents of src/core: {os.listdir(os.path.join(BASE_DIR, 'src', 'core')) if os.path.exists(os.path.join(BASE_DIR, 'src', 'core')) else 'Directory not found'}")


class LCRModels:
    def __init__(self, data_file='transaction_data.csv'):
        data_path = os.path.join(DATA_DIR, data_file)
        print(f"Attempting to read data from: {data_path}")
        self.df = pd.read_csv(data_path)
        # Try multiple common date formats
        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m/%d/%Y %H:%M:%S', '%m/%d/%Y']:
            self.df['date'] = pd.to_datetime(self.df['date'], format=fmt, errors='coerce')
            if not self.df['date'].isna().all():
                print(f"Successfully parsed 'date' with format: {fmt}")
                break
        if self.df['date'].isna().all():
            print("Sample of unparsed dates:", self.df['date'].head().tolist())
            raise ValueError("Failed to parse 'date' column. Check the date format. Sample dates printed above.")
        self.df.set_index('date', inplace=True)
        self.df.index.freq = 'D'  # Explicitly set daily frequency
        print(f"Data loaded, shape: {self.df.shape}, date range: {self.df.index.min()} to {self.df.index.max()}")
        # Feature engineering for LCR
        avg_outflow = self.df['outflows'].mean()
        self.df['net_outflow'] = self.df['outflows'].rolling(window=30, min_periods=30).sum() - self.df[
            'inflows'].rolling(window=30, min_periods=30).sum()
        self.df['net_outflow'] = np.where(self.df['net_outflow'] <= 0, avg_outflow * 0.05,
                                          self.df['net_outflow'])  # 5% threshold
        self.df['lcr'] = self.df['hqla_value'] / self.df['net_outflow'] * 100
        # Cap LCR outliers
        self.df['lcr'] = np.where(self.df['lcr'] > 1000, 1000, self.df['lcr'])
        self.df = self.df.dropna(subset=['lcr'])  # Remove rows with NaN LCR
        if len(self.df) < 30:
            print("Warning: Insufficient data for full 30-day LCR calculation; using available data")
        print(
            f"LCR variability: {self.df['lcr'].nunique()} unique values, min: {self.df['lcr'].min()}, max: {self.df['lcr'].max()}")
        # Fit scaler on the LCR data during initialization
        self.scaler = StandardScaler()
        self.scaler.fit(self.df['lcr'].values.reshape(-1, 1))
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.model_dir = MODEL_DIR
        self.test_predictions = None  # To store test set predictions
        # Ensure model is trained on initialization
        if not os.path.exists(os.path.join(self.model_dir, "best_lcr_model.pkl")):
            self.get_best_model()

    def prepare_lcr_data(self):
        series = self.df['lcr'].copy()
        print(f"Preparing LCR data, length: {len(series)}, min: {series.min()}, max: {series.max()}")
        return series

    def train_arima(self, series):
        print(f"Training ARIMA for LCR...")
        train_size = int(len(series) * 0.8)
        if train_size == len(series):
            train_size -= 1  # Ensure test set exists
        train, test = series[:train_size], series[train_size:]
        model = ARIMA(train, order=(1, 1, 1)).fit()
        forecast = model.forecast(len(test))
        mse = np.mean((test - forecast) ** 2) if len(test) > 0 else np.nan
        joblib.dump(model, os.path.join(self.model_dir, "arima_lcr_model.pkl"))
        self.test_predictions = pd.Series(forecast, index=test.index)
        print(f"LCR ARIMA MSE: {mse}")
        return model, mse

    def train_lstm(self, series):
        print(f"Training LSTM for LCR...")
        series_scaled = self.scaler.transform(series.values.reshape(-1, 1))
        X, y = self._create_sequences(series_scaled, look_back=30)
        if len(X) == 0:
            raise ValueError("Insufficient data for LSTM training after sequencing")
        train_size = int(len(X) * 0.8)
        if train_size == len(X):
            train_size -= 1
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        model = Sequential()
        model.add(LSTM(200, input_shape=(30, 1), return_sequences=True))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.1, verbose=0)
        y_pred = model.predict(X_test, verbose=0)
        mse = np.mean((y_test - y_pred) ** 2) if len(y_test) > 0 else np.nan
        joblib.dump(model, os.path.join(self.model_dir, "lstm_lcr_model.pkl"))
        # Inverse transform predictions to original scale
        y_pred_original = self.scaler.inverse_transform(y_pred)
        self.test_predictions = pd.Series(y_pred_original.flatten(), index=series.index[train_size + 30:len(series)])
        print(f"LCR LSTM MSE: {mse}")
        return model, mse

    def _create_sequences(self, data, look_back):
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:(i + look_back)])
            y.append(data[i + look_back])
        return np.array(X), np.array(y)

    def get_best_model(self):
        print("Determining best model for LCR...")
        lcr_series = self.prepare_lcr_data()
        if len(lcr_series) < 31:
            raise ValueError("Insufficient data for LCR modeling")
        arima_model, arima_mse = self.train_arima(lcr_series)
        lstm_model, lstm_mse = self.train_lstm(lcr_series)
        best_mse = min(arima_mse, lstm_mse)
        if np.isnan(best_mse):
            raise ValueError("No valid MSE calculated; check data variability or test set")
        if arima_mse <= lstm_mse:
            joblib.dump(arima_model, os.path.join(self.model_dir, "best_lcr_model.pkl"))
            print("Best model for LCR: ARIMA")
            return arima_model, "ARIMA"
        else:
            joblib.dump(lstm_model, os.path.join(self.model_dir, "best_lcr_model.pkl"))
            print("Best model for LCR: LSTM")
            return lstm_model, "LSTM"

    def forecast_lcr(self, steps=30):
        if os.path.exists(os.path.join(self.model_dir, "best_lcr_model.pkl")):
            model = joblib.load(os.path.join(self.model_dir, "best_lcr_model.pkl"))
            lcr_series = self.prepare_lcr_data()
            lcr_scaled = self.scaler.transform(lcr_series.values.reshape(-1, 1))
            look_back = 30

            if len(lcr_scaled) < look_back:
                raise ValueError(f"Insufficient data for sequence length {look_back}")

            last_sequence = lcr_scaled[-look_back:]
            forecast_values = []
            dates = pd.date_range(start=lcr_series.index[-1] + timedelta(days=1), periods=steps, freq='D')
            print(f"Forecast dates generated: {dates[0]} to {dates[-1]} for {steps} steps")  # Debug print

            if isinstance(model, ARIMA):
                forecast = model.forecast(steps)
                return pd.Series(forecast, index=dates)
            else:  # LSTM
                current_sequence = last_sequence.copy()
                for _ in range(steps):
                    current_sequence_reshaped = current_sequence.reshape(1, look_back, 1)
                    next_pred = model.predict(current_sequence_reshaped, verbose=0)
                    forecast_values.append(next_pred[0, 0])
                    current_sequence = np.roll(current_sequence, -1)
                    current_sequence[-1] = next_pred[0, 0]
                forecast_values = self.scaler.inverse_transform(np.array(forecast_values).reshape(-1, 1))
                return pd.Series(forecast_values.flatten(), index=dates)
        else:
            model, _ = self.get_best_model()
            return self.forecast_lcr(steps)


if __name__ == "__main__":
    print(f"Project root: {BASE_DIR}")
    model = LCRModels()
    model.get_best_model()
    lcr_forecast = model.forecast_lcr(steps=60)  # Test with 2 months
    print(
        f"Predicted LCR range: {lcr_forecast.min()} to {lcr_forecast.max()} on dates {lcr_forecast.index[0]} to {lcr_forecast.index[-1]}")
# LSTM Stock Price Prediction - Model Training Script

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
from datetime import date, timedelta


TICKER = "AAPL"
LOOKBACK = 60
TRAIN_SPLIT = 0.8
EPOCHS = 50
BATCH_SIZE = 32


print("Downloading AAPL data...")
start_date = date.today() - timedelta(days=365*2)  
end_date = date.today()

raw = yf.download(TICKER, start=start_date, end=end_date, auto_adjust=False)
if raw.empty:
    raise RuntimeError("No data downloaded!")


if 'Adj Close' in raw.columns:
    df = raw[['Adj Close']].rename(columns={'Adj Close': 'Close'})
elif 'Close' in raw.columns:
    df = raw[['Close']]
else:
    raise ValueError("No price column found!")

df = df.dropna()
print(f"Data shape: {df.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(df[["Close"]])


def make_sequences(series, window_size):
    X, y = [], []
    for i in range(window_size, len(series)):
        X.append(series[i-window_size:i])
        y.append(series[i])
    return np.array(X), np.array(y)

X, y = make_sequences(scaled_prices, LOOKBACK)
print(f"Sequences created: X shape {X.shape}, y shape {y.shape}")


split_idx = int(len(X) * TRAIN_SPLIT)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


print("Building LSTM model...")
model = models.Sequential([
    layers.Input(shape=(LOOKBACK, 1)),
    layers.LSTM(50, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(50, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(50),
    layers.Dropout(0.2),
    layers.Dense(25, activation="relu"),
    layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
print("Model compiled successfully!")

print("Training model...")
early_stop = callbacks.EarlyStopping(
    patience=10, 
    restore_best_weights=True, 
    monitor="val_loss"
)

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1,
    callbacks=[early_stop]
)


print("Evaluating model...")
pred_scaled = model.predict(X_test)
pred_prices = scaler.inverse_transform(pred_scaled)
actual_prices = scaler.inverse_transform(y_test)

rmse = np.sqrt(np.mean((pred_prices - actual_prices) ** 2))
mae = np.mean(np.abs(pred_prices - actual_prices))

print(f"Test RMSE: ${rmse:.2f}")
print(f"Test MAE: ${mae:.2f}")


model_filename = "lstm_stock_prediction_model.keras"
model.save(model_filename)
print(f"Model saved as: {model_filename}")


import joblib
joblib.dump(scaler, "scaler.pkl")
print("Scaler saved as: scaler.pkl")

print("\nTraining completed! You can now run your Streamlit app.")
print("To run the Streamlit app, use: streamlit run app.py")

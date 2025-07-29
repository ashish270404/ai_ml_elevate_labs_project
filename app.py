import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from datetime import date, timedelta
import plotly.graph_objects as go
import os  # Added import

TICKER, LOOKBACK = "AAPL", 60
MODEL_PATH = "lstm_stock_prediction_model.keras"  # Use relative path, not Windows path

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data(ttl=3600)
def fetch_data():
    start = date.today() - timedelta(days=365*5)
    data = yf.download(TICKER, start=start, end=date.today(), auto_adjust=False)
    if 'Adj Close' in data.columns:
        return data['Adj Close'].dropna()
    elif 'Close' in data.columns:
        return data['Close'].dropna()
    else:
        raise ValueError("Price column not found")

def forecast(series, model, scaler):
    window = series.values[-LOOKBACK:].reshape(LOOKBACK, 1)
    x = scaler.transform(window).reshape(1, LOOKBACK, 1)
    y_scaled = model.predict(x)[0][0]
    return float(scaler.inverse_transform([[y_scaled]])[0][0])

st.title(f"{TICKER} – LSTM Price Forecaster")

# Show current directory contents for debugging
st.write("Files in current directory:", os.listdir("."))

# Load model and data
model = load_model()
prices = fetch_data()
prices = prices.squeeze()

# Fit scaler
scaler = MinMaxScaler().fit(prices.values.reshape(-1, 1))

# Predict next close price
next_close = forecast(prices, model, scaler)
last_close = prices.iloc[-1]

# Show metrics
st.metric("Last Close", f"${last_close:.2f}")
st.metric("Next-Day Forecast", f"${next_close:.2f}", delta=f"${next_close - last_close:+.2f}")

# Price chart
st.line_chart(prices.tail(250))

fig = go.Figure()
fig.add_trace(go.Scatter(x=prices.index, y=prices, name="Adj Close"))
fig.update_layout(height=400, margin=dict(t=30, b=20))
st.plotly_chart(fig, use_container_width=True)



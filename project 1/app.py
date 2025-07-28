import streamlit as st, yfinance as yf, numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from datetime import date, timedelta
import plotly.graph_objects as go

TICKER, LOOKBACK = "AAPL", 60
MODEL_PATH = "lstm_stock_prediction_model.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data(ttl=3600)
def fetch_data():
    start = date.today() - timedelta(days=365*5)
    return yf.download(TICKER, start=start)["Adj Close"].dropna()

def forecast(series, model, scaler):
    window = series.values[-LOOKBACK:].reshape(LOOKBACK,1)
    x = scaler.transform(window).reshape(1,LOOKBACK,1)
    y_scaled = model.predict(x)[0][0]
    return float(scaler.inverse_transform([[y_scaled]])[0][0])

st.title(f"{TICKER} – LSTM Price Forecaster")

model = load_model()
prices = fetch_data()
scaler = MinMaxScaler().fit(prices.values.reshape(-1,1))
next_close = forecast(prices, model, scaler)

st.line_chart(prices.tail(250))
st.metric("Last Close", f"${prices.iloc[-1]:.2f}")
st.metric("Next-Day Forecast", f"${next_close:.2f}",
          delta=f"${next_close - prices.iloc[-1]:+.2f}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=prices.index, y=prices, name="Adj Close"))
fig.update_layout(height=400, margin=dict(t=30, b=20))
st.plotly_chart(fig, use_container_width=True)

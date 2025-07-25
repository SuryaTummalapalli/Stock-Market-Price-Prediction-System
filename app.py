import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# --- Streamlit App UI ---
st.set_page_config(page_title="Stock Market Price Prediction", layout="wide")
st.markdown(
    """
    <style>
        body {
            background-color: #0e1117;
        }
        .stApp {
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title ---
st.title("📈 Stock Market Price Prediction System")

# --- Load Ticker Symbols ---
try:
    url = "https://www1.nseindia.com/content/indices/ind_nifty50list.csv"
    df = pd.read_csv(url, on_bad_lines='skip')
    nifty_tickers = df["Symbol"].tolist()
except Exception as e:
    st.warning(f"Could not fetch Nifty 50 tickers. Falling back to default list. Error: {e}")
    nifty_tickers = ["AAPL", "TCS.NS", "INFY.NS", "GOOGL", "MSFT"]

# --- Input Section Layout ---
col1, col2 = st.columns(2)

with col1:
    default_tickers = nifty_tickers
    custom_ticker = st.text_input("Enter a custom stock ticker (e.g., RELIANCE.NS, TSLA):").strip().upper()
    if custom_ticker:
        selected_ticker = custom_ticker
    else:
        selected_ticker = st.selectbox("Or select from Nifty 50 stocks:", default_tickers)

with col2:
    today = datetime.today()
    start_date = st.date_input("Start Date", today - timedelta(days=180), max_value=today - timedelta(days=1))
    end_date = st.date_input("End Date", today, min_value=start_date + timedelta(days=1), max_value=today)

if start_date >= end_date:
    st.error("Error: End date must be after start date.")
    st.stop()

# --- Model Selection and Parameters ---
with st.expander("Model & Prediction Parameters", expanded=True):
    model_choice = st.radio("Choose Prediction Model:", ("Random Forest", "LSTM"))

    if model_choice == "Random Forest":
        rf_lag = st.slider("Random Forest Lag Days:", 1, 30, 5)
    elif model_choice == "LSTM":
        lstm_epochs = st.slider("LSTM Epochs:", 10, 100, 20)
        lstm_batch_size = st.selectbox("LSTM Batch Size:", [16, 32, 64], index=1)

# --- Placeholder for Future Data Fetching and Modeling ---
st.write(f"Selected Ticker: `{selected_ticker}` from `{start_date}` to `{end_date}`")
st.info("⚠️ Model logic not yet implemented. This is just the UI layout and ticker fix.")


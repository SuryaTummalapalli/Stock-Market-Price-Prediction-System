import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import LSTM, Dense

import plotly.graph_objs as go
import io
import matplotlib.pyplot as plt

# --- Helper Functions ---

def create_lag_features(df, lag_days=5):
    for i in range(1, lag_days + 1):
        df[f'lag_{i}'] = df['Close'].shift(i)
    df = df.dropna()
    return df

def prepare_lstm_data(series, time_step=60):
    X, y = [], []
    for i in range(time_step, len(series)):
        X.append(series[i - time_step:i])
        y.append(series[i])
    X, y = np.array(X), np.array(y)
    return X.reshape((X.shape[0], X.shape[1], 1)), y

def predict_next_days_rf(rf_model, last_data, n_days):
    preds = []
    current_input = np.array(last_data).copy()
    for _ in range(n_days):
        pred = rf_model.predict(current_input.reshape(1, -1))[0]
        preds.append(pred)
        current_input = np.roll(current_input, -1)
        current_input[-1] = pred
    return preds

def predict_next_days_lstm(model, input_seq, n_days, scaler):
    preds = []
    current_seq = np.array(input_seq).copy()
    for _ in range(n_days):
        pred = model.predict(current_seq.reshape(1, current_seq.shape[0], 1), verbose=0)[0][0]
        preds.append(pred)
        current_seq = np.append(current_seq[1:], pred)
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds

def get_next_n_business_days(last_date, n):
    dates = []
    current = last_date
    while len(dates) < n:
        current += timedelta(days=1)
        if current.weekday() < 5:
            dates.append(current)
    return dates

# --- Cached Functions ---

@st.cache_data(show_spinner=False)
def load_stock_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)

@st.cache_resource(show_spinner=False)
def train_rf_model(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf

@st.cache_resource(show_spinner=False)
def train_lstm_model(X_train, y_train, time_step):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    return model

# --- Streamlit App ---

st.title("📈 Stock Market Price Prediction System")

# Layout for inputs
col1, col2 = st.columns(2)

with col1:
    default_tickers = ["AAPL", "TCS.NS", "INFY.NS", "GOOGL", "MSFT"]
    custom_ticker = st.text_input("Enter a custom stock ticker (e.g., RELIANCE.NS, TSLA):").strip().upper()
    if custom_ticker:
        selected_ticker = custom_ticker
    else:
        selected_ticker = st.selectbox("Or select from default stocks:", default_tickers)

with col2:
    today = datetime.today()
    start_date = st.date_input("Start Date", today - timedelta(days=180), max_value=today - timedelta(days=1))
    end_date = st.date_input("End Date", today, min_value=start_date + timedelta(days=1), max_value=today)

if start_date >= end_date:
    st.error("Error: End date must be after start date.")
    st.stop()

# Model and prediction parameters in expander
with st.expander("Model & Prediction Parameters", expanded=True):
    model_choice = st.radio("Choose Prediction Model:", ["Random Forest", "LSTM"])
    if model_choice == "LSTM":
        time_step = st.slider("LSTM Time Step (lookback days):", min_value=10, max_value=90, value=60, help="Number of past days for each prediction.")
    else:
        time_step = st.slider("Random Forest Lag Days:", min_value=1, max_value=20, value=5, help="Number of lag days as features.")
    n_days = st.slider("Predict Next N Business Days:", min_value=1, max_value=14, value=3)

# --- Data Pipeline: Fetch historical stock price data (Open, High, Low, Close, Volume) ---

from datetime import datetime
last_updated = datetime.now().strftime('%d %b %Y, %H:%M')
st.info(f"Fetching data for **{selected_ticker}** from {start_date} to {end_date} ... Last updated at {last_updated}")
stock_data = load_stock_data(selected_ticker, start_date, end_date)
if stock_data.empty:
    st.error("No data fetched. Check ticker and date range.")
    st.stop()

# Show the first 10 rows of the full OHLCV data
st.write("Sample historical stock data (OHLCV):")
st.dataframe(stock_data.head(10))

# For modeling, use only the 'Close' column as before
df = stock_data[['Close']].copy().dropna()
df['MA20'] = df['Close'].rolling(window=20).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()
st.dataframe(df.tail(10).style.format({'Close': '{:.2f}', 'MA20': '{:.2f}', 'MA50': '{:.2f}'}))

# --- Model Training and Prediction ---

if model_choice == "Random Forest":
    st.subheader("Random Forest Regressor")
    df_rf = create_lag_features(df.copy(), time_step)
    if df_rf.empty:
        st.error(f"Not enough data for Random Forest with lag days = {time_step}. Please increase date range or reduce lag days.")
        st.stop()
    X = df_rf[[f'lag_{i}' for i in range(1, time_step + 1)]].values
    y = df_rf['Close'].values
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    rf_model = train_rf_model(X_train, y_train)
    
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    st.info(f"Train RMSE: {train_rmse:.2f} | Test RMSE: {test_rmse:.2f}")
    
    last_data = df['Close'].values[-time_step:]
    preds = predict_next_days_rf(rf_model, last_data, n_days)
    
elif model_choice == "LSTM":
    st.subheader("LSTM Model")
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_prices = df['Close'].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(close_prices)

    if len(scaled_data) < time_step:
        st.error(f"Not enough data for LSTM with time step {time_step}.")
        st.stop()

    X_lstm, y_lstm = prepare_lstm_data(scaled_data, time_step=time_step)
    
    split_idx = int(len(X_lstm) * 0.8)
    X_train, X_test = X_lstm[:split_idx], X_lstm[split_idx:]
    y_train, y_test = y_lstm[:split_idx], y_lstm[split_idx:]
    
    with st.spinner("Training LSTM model..."):
        lstm_model = train_lstm_model(X_train, y_train, time_step)
    
    y_pred_train = lstm_model.predict(X_train, verbose=0)
    y_pred_test = lstm_model.predict(X_test, verbose=0)
    
    train_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_train.reshape(-1,1)), scaler.inverse_transform(y_pred_train)))
    test_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test.reshape(-1,1)), scaler.inverse_transform(y_pred_test)))
    
    st.info(f"Train RMSE: {train_rmse:.2f} | Test RMSE: {test_rmse:.2f}")
    
    last_sequence = scaled_data[-time_step:].flatten()
    preds = predict_next_days_lstm(lstm_model, last_sequence, n_days, scaler)
else:
    st.error("Invalid model choice.")
    st.stop()

last_date = df.index[-1]
pred_dates = get_next_n_business_days(last_date, n_days)
pred_df = pd.DataFrame({'Date': pred_dates, 'Predicted Close': preds}).set_index('Date')

# --- Visualization ---

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Historical Close', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA20', line=dict(color='green', dash='dash')))
fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='MA50', line=dict(color='purple', dash='dot')))
fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Predicted Close'], mode='lines+markers', name='Predicted Close', line=dict(color='orange')))
fig.update_layout(title=f"Stock Price Prediction for {selected_ticker}", xaxis_title='Date', yaxis_title='Price', hovermode='x unified')

st.plotly_chart(fig, use_container_width=True)

# Matplotlib Visualization for reference
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Close'], label='Historical Close', color='blue')
plt.plot(pred_df.index, pred_df['Predicted Close'], label='Predicted Close', color='orange')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f"Stock Price Prediction for {selected_ticker} (Matplotlib)")
plt.legend()
plt.tight_layout()
st.pyplot(plt)

# --- Summary ---
final_pred_price = preds[-1]
st.markdown(f"### Summary: {model_choice} predicts {selected_ticker} will close at **${final_pred_price:.2f}** in {n_days} business days.")

# --- Download CSVs ---

combined_df = pd.concat([
    df[['Close']].rename(columns={'Close': 'Actual Close'}),
    pred_df.rename(columns={'Predicted Close': 'Predicted Close'})
]).reset_index()

csv_buffer = io.StringIO()
combined_df.to_csv(csv_buffer, index=False)
csv_buffer.seek(0)
st.download_button(
    label="Download Historical + Predicted Prices as CSV",
    data=csv_buffer.getvalue(),
    file_name=f"{selected_ticker}_price_predictions.csv",
    mime="text/csv"
)

pred_csv_buffer = io.StringIO()
pred_df.reset_index().to_csv(pred_csv_buffer, index=False)
pred_csv_buffer.seek(0)
st.download_button(
    label="Download Only Predictions as CSV",
    data=pred_csv_buffer.getvalue(),
    file_name=f"{selected_ticker}_predictions_only.csv",
    mime="text/csv"
)


# --- Styled Footer ---
st.markdown(""
<style>
.footer {
    position: relative;
    bottom: 0;
    width: 100%;
    text-align: center;
    color: gray;
    font-size: 14px;
    margin-top: 50px;
    padding: 10px 0;
    border-top: 1px solid #ddd;
}




# --- Footer / Watermark ---

# --- End of Streamlit App ---

import gradio as gr
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#  Load Model & Scaler 
MODEL_PATH = "lstm_stock_forecast_full.h5"
SCALER_PATH = "lstm_full_scaler.joblib"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

# Config 
FEATURES = ['Close', 'Open', 'High', 'Low', 'Volume']
TARGET_NAME = 'Close'
WINDOW = 60

# Helper Functions 
def inverse_target(y_scaled, scaler, target_col):
    """Inverse-transform only target column."""
    zeros = np.zeros((len(y_scaled), len(FEATURES)))
    tgt_idx = FEATURES.index(target_col)
    zeros[:, tgt_idx] = y_scaled
    inv = scaler.inverse_transform(zeros)
    return inv[:, tgt_idx]

def predict_future(days_ahead: int = 1):
    """Fetch last 90 days of TSLA data, forecast next N days using recursive LSTM."""
    data = yf.download("TSLA", period="90d", interval="1d", progress=False, auto_adjust=False)

    # Flatten MultiIndex if needed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]

    if 'Close' not in data.columns:
        return None, "❌ Could not find 'Close' column in downloaded data."

    data = data.dropna(subset=['Close'])

    df = data[FEATURES].copy()
    scaled = pd.DataFrame(scaler.transform(df), columns=FEATURES, index=df.index)

    seq = scaled.tail(WINDOW).values.reshape(1, WINDOW, len(FEATURES))
    predictions_scaled = []
    last_seq = seq.copy()

    for _ in range(days_ahead):
        yhat_scaled = model.predict(last_seq, verbose=0).squeeze().item()
        predictions_scaled.append(yhat_scaled)

        next_step = last_seq[0, -1, :].copy()
        tgt_idx = FEATURES.index(TARGET_NAME)
        next_step[tgt_idx] = yhat_scaled
        last_seq = np.concatenate([last_seq[:, 1:, :], next_step.reshape(1, 1, len(FEATURES))], axis=1)

    preds = inverse_target(np.array(predictions_scaled), scaler, TARGET_NAME)

    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days_ahead)
    pred_df = pd.DataFrame({'Predicted Price': preds}, index=future_dates)

    plt.figure(figsize=(10, 5))
    plt.plot(df.index[-60:], df['Close'].iloc[-60:], label="Last 60 Days", color='blue')
    plt.plot(pred_df.index, pred_df['Predicted Price'], label=f"Next {days_ahead} Days Forecast", color='red', marker='o')
    plt.title(f"Tesla Stock Forecast ({days_ahead}-Day Ahead)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    return plt, f"Predicted price on {future_dates[-1].date()}: ${preds[-1]:.2f}"


#  Gradio UI 
with gr.Blocks(title="Tesla LSTM Stock Forecast") as demo:
    gr.Markdown("## 🚗 Tesla Stock Forecast (LSTM Model)\nPredict next **N-day** closing prices for Tesla (TSLA).")
    days = gr.Slider(minimum=1, maximum=30, value=5, step=1, label="Days Ahead to Forecast")
    run = gr.Button("🔮 Predict")
    plot = gr.Plot(label="Forecast Plot")
    text = gr.Textbox(label="Forecast Summary")
    run.click(fn=predict_future, inputs=days, outputs=[plot, text])

demo.launch()

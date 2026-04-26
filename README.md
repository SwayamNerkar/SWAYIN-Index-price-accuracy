# 📈 AI-Based Stock Price Prediction System using LSTM

> **Deep Learning · Real-Time Data · BUY/SELL Signals · Interactive Dashboard**

---

## 🎯 Project Overview

A **production-ready** stock price prediction system that uses a **Deep Stacked LSTM** neural network to:
- Predict **next-day** and **next 5-minute** NIFTY 50 / any stock prices
- Generate actionable **BUY / SELL signals** with confidence levels
- Provide a stunning **Streamlit interactive dashboard** with live Plotly charts
- Run full **backtesting** to validate strategy performance

---

## 📁 Project Structure

```
stock-ai-project/
│
├── main.py              # 🚀 Master orchestrator (run this!)
├── data_fetcher.py      # 📡 Robust data fetching (yfinance → Alpha Vantage)
├── features.py          # 🔬 Technical indicator engineering
├── model.py             # 🧠 Deep LSTM model builder & trainer
├── predictor.py         # 🎯 Prediction engine (next-day, 5-min simulation)
├── utils.py             # 🛠️  Logging, metrics, backtesting, signal generator
├── dashboard.py         # 📊 Streamlit interactive dashboard
├── config.py            # ⚙️  All hyperparameters & settings
├── requirements.txt     # 📦 Python dependencies
├── model.h5             # 💾 Saved trained model (auto-generated)
├── advanced_prediction.png  # 🖼️ Chart output (auto-generated)
├── prediction_result.json   # 📋 JSON results (auto-generated)
└── stock_ai.log         # 📜 Log file (auto-generated)
```

---

## ⚙️ Tech Stack

| Category | Libraries |
|---|---|
| Deep Learning | TensorFlow / Keras |
| Data Science | Pandas, NumPy, Scikit-learn |
| Visualization | Matplotlib, Plotly |
| Data APIs | yfinance (Yahoo Finance — no key needed) |
| Dashboard | Streamlit |
| Language | Python 3.9+ |

---

## 🧠 Model Architecture

```
Input: (60 timesteps × N features)
        ↓
LSTM(100, return_sequences=True)
Dropout(0.2)
        ↓
LSTM(100, return_sequences=True)
Dropout(0.2)
        ↓
LSTM(50, return_sequences=False)
Dropout(0.2)
        ↓
Dense(25, relu)
        ↓
Dense(1)  ← Predicted Close Price
```

- **Optimizer:** Adam
- **Loss:** Mean Squared Error
- **Epochs:** 50 (with EarlyStopping)
- **Batch Size:** 32
- **Look-back window:** 60 time steps

---

## 🔬 Features Engineered

| Indicator | Description |
|---|---|
| **MA50** | Simple Moving Average (50-period) |
| **EMA50** | Exponential Moving Average (50-period) |
| **RSI14** | Relative Strength Index (14-period) |
| **Bollinger Bands** | Upper / Middle / Lower (20-period) |
| **MACD** | MACD Line, Signal Line, Histogram |
| **ATR** | Average True Range (14-period) |
| **Volume** | Raw trading volume |

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the Main Pipeline

```bash
# Default: NIFTY 50 daily prediction
python main.py

# Custom stock
python main.py --symbol AAPL

# 5-minute intraday
python main.py --symbol ^NSEI --interval 5m

# Force retrain even if model.h5 exists
python main.py --retrain
```

### Step 3: Launch Streamlit Dashboard (Optional)

```bash
streamlit run dashboard.py
```

Open browser at: **http://localhost:8501**

---

## 📡 Data Fetching Logic

```
fetch_stock_data(symbol, interval, period)
    │
    ├── Attempt 1 of 3: yfinance
    │     ✅ Success → clean DataFrame returned
    │     ❌ Failure → log warning, retry
    │
    ├── Attempt 2 of 3: yfinance (retry)
    │     ✅ Success → clean DataFrame returned
    │     ❌ Failure → retry once more
    │
    └── Attempt 3 of 3: yfinance (final retry)
          ✅ Success → clean DataFrame returned
          ❌ Failure → raise RuntimeError with details
```

**Data source:** yfinance (Yahoo Finance) — free, no API key needed ✅

---

## 📊 Sample Output

```
======================================================
  AI STOCK PREDICTOR — ^NSEI  [1d]
======================================================
[yfinance] ✅ Success — 1825 rows fetched for ^NSEI.
Feature engineering complete: 1771 rows retained.
Preprocessing: X_train=(1357, 60, 11) | X_test=(354, 60, 11)
Training complete. Best val_loss: 0.000213

======================================================
  FINAL PREDICTION REPORT
======================================================
  Symbol                  : ^NSEI
  Current Price           : ₹ 22,345.65

  ── Next-Day Forecast ──
  Predicted Next-Day Close  : ₹ 22,498.10
  Signal                    : BUY 📈
  Confidence (%)            : 68.3

  ── 5-Min Simulation ──
  t+5min                    : ₹ 22,352.10
  t+10min                   : ₹ 22,358.90
  t+15min                   : ₹ 22,365.40
  t+20min                   : ₹ 22,372.80
  t+25min                   : ₹ 22,381.20

  ── Evaluation ──
  RMSE                      : 145.2300
  MAE                       : 98.7100
  MAPE (%)                  : 0.6520
  Directional Accuracy      : 58.4000

  ── Backtesting ──
  Initial Capital           : ₹ 100,000
  Final Capital             : ₹ 134,250
  Total Return (%)          : 34.25
  Number of Trades          : 47
  Win Rate (%)              : 61.7
======================================================
```

---

## 📈 Signal Logic

```python
if predicted_price > current_price:
    signal = "BUY 📈"
    confidence = min(abs(change_pct) * 10, 99.0)
else:
    signal = "SELL 📉"
    confidence = min(abs(change_pct) * 10, 99.0)
```

---

## 🔧 Configuration (config.py)

All parameters are centralized in `config.py`:

```python
DEFAULT_SYMBOL   = "^NSEI"    # Change stock here
EPOCHS           = 50
BATCH_SIZE       = 32
TIME_STEP        = 60          # Look-back window
LSTM_UNITS       = [100, 100, 50]
DROPOUT_RATE     = 0.2
TRAIN_RATIO      = 0.80
```

---

## 🧪 Backtesting Strategy

- **Entry:** BUY when `predicted_price > current_price`
- **Exit:** SELL when `predicted_price < current_price`
- **Transaction cost:** 0.1% per trade
- **Metrics:** Total Return %, Win Rate, Number of Trades

---

## 🌐 Multi-Stock Support

```python
from data_fetcher import fetch_multiple_stocks

results = fetch_multiple_stocks(
    symbols=["^NSEI", "RELIANCE.NS", "TCS.NS", "AAPL"],
    interval="1d",
    period="3y",
)
```

---

## 📋 Logs

All activity is logged to:
- **Console** (INFO level)
- **`stock_ai.log`** (DEBUG level, full detail)

---

## ❌ Error Handling

| Scenario | Handling |
|---|---|
| Network / connection error | Auto-retry 3× with 3s delay between attempts |
| Empty DataFrame returned | `ValueError` with clear message |
| Invalid ticker symbol | `ValueError` — check Yahoo Finance for correct format |
| Model file missing | Automatically retrains from scratch |
| Insufficient data (<100 rows) | Validation error with explanation |

---

## 📦 Output Files

| File | Description |
|---|---|
| `model.h5` | Trained Keras LSTM model |
| `advanced_prediction.png` | 6-panel prediction chart |
| `prediction_result.json` | Full prediction results as JSON |
| `stock_ai.log` | Detailed execution log |

---

## 👨‍💻 Academic Submission Notes

This project demonstrates:
1. **Data Engineering** — Robust yfinance data pipeline with retries & validation
2. **Feature Engineering** — 6+ technical indicators (RSI, MACD, Bollinger, ATR)
3. **Deep Learning** — 3-layer stacked LSTM with Dropout regularisation
4. **Model Evaluation** — RMSE, MAE, MAPE, Directional Accuracy
5. **Backtesting** — Realistic trading simulation with transaction costs
6. **Production Code** — Logging, config, error handling, modular design
7. **Visualisation** — Matplotlib static + Plotly interactive dashboard

---

## 🛡️ License

For academic/educational use. All market data is fetched from public APIs.

---

*Built with ❤️ using TensorFlow, yfinance, and Streamlit*
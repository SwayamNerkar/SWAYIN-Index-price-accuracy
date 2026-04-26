"""
config.py - Centralized Configuration File
==========================================
All project parameters and settings are managed here.
Modify this file to tune the model, change stocks, or adjust training settings.

Data Source: yfinance (Yahoo Finance) — no API key required.
"""

# ─────────────────────────────────────────────
#  STOCK / INDEX SETTINGS
# ─────────────────────────────────────────────
# Default symbol (Yahoo Finance format for NIFTY 50)
DEFAULT_SYMBOL   = "^NSEI"
DEFAULT_INTERVAL = "1d"          # '1d' for daily, '5m' for 5-minute
DEFAULT_PERIOD   = "5y"          # Historical period for daily data

# Multi-stock support: add tickers here for batch analysis
MULTI_STOCK_LIST = ["^NSEI", "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]

# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────
MA_WINDOW  = 50    # Simple Moving Average window
EMA_WINDOW = 50    # Exponential Moving Average window
RSI_WINDOW = 14    # RSI period

# ─────────────────────────────────────────────
#  PREPROCESSING
# ─────────────────────────────────────────────
TIME_STEP   = 100  # Sequence length (look-back window)
TRAIN_RATIO = 0.80 # 80% train / 20% test split

# ─────────────────────────────────────────────
#  MODEL ARCHITECTURE
# ─────────────────────────────────────────────
LSTM_UNITS   = [128, 128, 64]  # Increased units per LSTM layer
DROPOUT_RATE = 0.15           # Slightly reduced dropout for better fit
DENSE_UNITS  = 1               # Output: single price prediction

# ─────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────
EPOCHS     = 25
BATCH_SIZE = 16
OPTIMIZER  = "adam"
LOSS       = "huber_loss"      # Huber loss is more robust to outliers than MSE

# ─────────────────────────────────────────────
#  FILE PATHS
# ─────────────────────────────────────────────
MODEL_PATH = "model.h5"
PLOT_PATH  = "advanced_prediction.png"
LOG_FILE   = "stock_ai.log"

# ─────────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────────
NEXT_5MIN_STEPS = 5   # Number of 5-minute steps to simulate ahead

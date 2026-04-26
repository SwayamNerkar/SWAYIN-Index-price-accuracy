"""
features.py - Feature Engineering Module
==========================================
Adds the following technical indicators to a raw OHLCV DataFrame:
  - MA50   : Simple Moving Average (50-day)
  - EMA50  : Exponential Moving Average (50-day)
  - RSI14  : Relative Strength Index (14-day)
  - Volume : Already present; kept as informational feature

All NaN rows produced by windowed indicators are dropped cleanly.
"""

import numpy as np
import pandas as pd
from utils import logger
from config import MA_WINDOW, EMA_WINDOW, RSI_WINDOW


# ─────────────────────────────────────────────
#  INDIVIDUAL INDICATOR CALCULATORS
# ─────────────────────────────────────────────
def add_moving_average(df: pd.DataFrame, window: int = MA_WINDOW) -> pd.DataFrame:
    """Add Simple Moving Average (SMA) column."""
    col = f"MA{window}"
    df[col] = df["Close"].rolling(window=window).mean()
    logger.debug(f"Added {col} indicator.")
    return df


def add_ema(df: pd.DataFrame, window: int = EMA_WINDOW) -> pd.DataFrame:
    """Add Exponential Moving Average (EMA) column."""
    col = f"EMA{window}"
    df[col] = df["Close"].ewm(span=window, adjust=False).mean()
    logger.debug(f"Added {col} indicator.")
    return df


def add_rsi(df: pd.DataFrame, window: int = RSI_WINDOW) -> pd.DataFrame:
    """
    Add RSI (Relative Strength Index) using Wilder's smoothing method.
    Values range from 0 to 100.
      > 70 → Overbought
      < 30 → Oversold
    """
    col   = f"RSI{window}"
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    # Wilder's smoothing (equivalent to EMA with alpha=1/window)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs        = avg_gain / avg_loss.replace(0, np.nan)
    df[col]   = 100 - (100 / (1 + rs))
    logger.debug(f"Added {col} indicator.")
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Add Bollinger Bands (Upper, Middle, Lower).
    Bonus indicator for advanced analysis.
    """
    middle         = df["Close"].rolling(window=window).mean()
    std            = df["Close"].rolling(window=window).std()
    df["BB_Upper"] = middle + 2 * std
    df["BB_Middle"]= middle
    df["BB_Lower"] = middle - 2 * std
    logger.debug("Added Bollinger Bands (BB_Upper, BB_Middle, BB_Lower).")
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add MACD (Moving Average Convergence Divergence).
    MACD Line = EMA12 - EMA26
    Signal    = EMA9 of MACD Line
    Histogram = MACD - Signal
    """
    ema12           = df["Close"].ewm(span=12, adjust=False).mean()
    ema26           = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]      = ema12 - ema26
    df["MACD_Signal"]= df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    logger.debug("Added MACD indicators.")
    return df


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Average True Range — measures volatility.
    ATR = EMA(True Range, window)
    True Range = max(H-L, |H-prev_C|, |L-prev_C|)
    """
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)

    df["ATR"] = tr.ewm(span=window, adjust=False).mean()
    logger.debug("Added ATR indicator.")
    return df


# ─────────────────────────────────────────────
#  MASTER FEATURE PIPELINE
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline on a raw OHLCV DataFrame.

    Adds:
      MA50, EMA50, RSI14, Bollinger Bands, MACD, ATR

    Then drops all rows with NaN to produce a clean, model-ready DataFrame.

    Args:
        df      : Raw OHLCV DataFrame (columns: Open, High, Low, Close, Volume)
        verbose : Log shape info if True

    Returns:
        Feature-enriched DataFrame (NaN rows removed).
    """
    if df is None or df.empty:
        raise ValueError("Cannot engineer features on an empty DataFrame.")

    original_len = len(df)
    df = df.copy()

    # ── Core required indicators ──────────────────────────────
    df = add_moving_average(df, window=MA_WINDOW)
    df = add_ema(df, window=EMA_WINDOW)
    df = add_rsi(df, window=RSI_WINDOW)

    # ── Bonus advanced indicators ─────────────────────────────
    df = add_bollinger_bands(df, window=20)
    df = add_macd(df)
    df = add_atr(df, window=14)

    # ── Drop NaN rows (produced by rolling/ewm windows) ───────
    df = df.dropna()

    if verbose:
        dropped = original_len - len(df)
        logger.info(
            f"Feature engineering complete: "
            f"{len(df)} rows retained ({dropped} NaN rows dropped). "
            f"Columns: {list(df.columns)}"
        )

    return df


# ─────────────────────────────────────────────
#  QUICK SANITY CHECK (run directly)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from data_fetcher import fetch_stock_data
    raw  = fetch_stock_data()
    rich = engineer_features(raw)
    print(rich.tail(10))
    print("\nShape:", rich.shape)
    print("Columns:", rich.columns.tolist())

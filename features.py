"""
features.py - Backward Compatibility Adapter for Backend Feature Pipeline
"""

import pandas as pd
from backend.app.feature_engineering.pipeline import run_feature_pipeline
from backend.app.feature_engineering.technical import (
    calculate_technical_indicators,
    calculate_technical_indicators as engineer_features_tech
)

def add_moving_average(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    df[f"MA{window}"] = df["Close"].rolling(window=window).mean()
    return df

def add_ema(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    df[f"EMA{window}"] = df["Close"].ewm(span=window, adjust=False).mean()
    return df

def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    df[f"RSI{window}"] = 100 - (100 / (1 + rs))
    return df

def add_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    m = df["Close"].rolling(window=window).mean()
    s = df["Close"].rolling(window=window).std()
    df["BB_Upper"] = m + 2 * s
    df["BB_Middle"] = m
    df["BB_Lower"] = m - 2 * s
    return df

def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df

def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=window, adjust=False).mean()
    return df

def engineer_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Run backend feature pipeline.
    """
    return run_feature_pipeline(df, drop_na=True)

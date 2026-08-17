import numpy as np
import pandas as pd
from backend.app.feature_engineering.technical import calculate_technical_indicators
from backend.app.core.logging import logger

def add_lag_features(df: pd.DataFrame, lags: list = [1, 2, 3, 5]) -> pd.DataFrame:
    for lag in lags:
        df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)
        df[f"Return_Lag_{lag}"] = df["Returns"].shift(lag)
        df[f"RSI_Lag_{lag}"] = df["RSI14"].shift(lag)
    return df

def calculate_support_resistance(df: pd.DataFrame, window: int = 20) -> dict:
    """
    Calculate Pivot Points, Support levels (S1, S2, S3), and Resistance levels (R1, R2, R3).
    """
    recent = df.tail(window)
    high = recent["High"].max()
    low = recent["Low"].min()
    close = recent["Close"].iloc[-1]

    pivot = (high + low + close) / 3.0
    r1 = (2 * pivot) - low
    s1 = (2 * pivot) - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)

    return {
        "Pivot": round(float(pivot), 2),
        "S1": round(float(s1), 2),
        "S2": round(float(s2), 2),
        "S3": round(float(s3), 2),
        "R1": round(float(r1), 2),
        "R2": round(float(r2), 2),
        "R3": round(float(r3), 2),
        "Recent_High": round(float(high), 2),
        "Recent_Low": round(float(low), 2)
    }

def detect_market_regime(df: pd.DataFrame) -> dict:
    """
    Detect market regime based on ADX, EMA50, EMA200, and Volatility:
      - Bullish Trend
      - Bearish Trend
      - High Volatility Breakout
      - Low Volatility Consolidation
    """
    latest = df.iloc[-1]
    close = latest["Close"]
    ema50 = latest.get("EMA_50", close)
    ema200 = latest.get("EMA_200", close)
    adx = latest.get("ADX", 20.0)
    volatility = latest.get("Rolling_Volatility", 0.15)

    if close > ema50 and ema50 > ema200 and adx >= 25:
        regime = "Bullish Strong Trend"
    elif close < ema50 and ema50 < ema200 and adx >= 25:
        regime = "Bearish Strong Trend"
    elif volatility > 0.25:
        regime = "High Volatility Expansion"
    else:
        regime = "Consolidation / Range-Bound"

    return {
        "Regime": regime,
        "ADX": round(float(adx), 2),
        "Volatility": round(float(volatility), 4),
        "Above_EMA200": bool(close > ema200)
    }

def run_feature_pipeline(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """
    Full feature pipeline execution.
    """
    logger.info("Executing advanced Feature Engineering pipeline...")
    df = calculate_technical_indicators(df)
    df = add_lag_features(df)
    
    if drop_na:
        df = df.dropna().copy()
        
    logger.info(f"Feature engineering pipeline completed. Total features: {len(df.columns)}, Total rows: {len(df)}")
    return df

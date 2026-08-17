import numpy as np
import pandas as pd
from backend.app.core.logging import logger

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate full technical indicator suite on OHLCV DataFrame:
    EMA, SMA, VWAP, RSI, MACD, ATR, ADX, CCI, ROC, Momentum,
    Williams %R, Stochastic RSI, Bollinger Bands, Ichimoku Cloud,
    SuperTrend, OBV, CMF, MFI, Rolling Stats, Price Change %, Returns.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty.")
        
    df = df.copy()
    
    # Standardize column names
    col_map = {c: c.capitalize() for c in df.columns}
    df.rename(columns=col_map, inplace=True)
    
    for req in ["Open", "High", "Low", "Close", "Volume"]:
        if req not in df.columns:
            raise ValueError(f"Missing required column: {req}")

    n_rows = len(df)

    # 1. Moving Averages (using min_periods=1 to preserve data on short periods)
    df["SMA_20"] = df["Close"].rolling(window=min(20, n_rows), min_periods=1).mean()
    df["SMA_50"] = df["Close"].rolling(window=min(50, n_rows), min_periods=1).mean()
    df["SMA_200"] = df["Close"].rolling(window=min(200, n_rows), min_periods=1).mean()
    
    df["EMA_9"] = df["Close"].ewm(span=min(9, n_rows), adjust=False).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=min(50, n_rows), adjust=False).mean()
    df["EMA_200"] = df["Close"].ewm(span=min(200, n_rows), adjust=False).mean()
    
    # Compatibility column aliases
    df["MA50"] = df["SMA_50"]
    df["EMA50"] = df["EMA_50"]

    # 2. VWAP (Volume Weighted Average Price)
    cum_vol = df["Volume"].cumsum()
    cum_vol_price = (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3.0).cumsum()
    df["VWAP"] = np.where(cum_vol != 0, cum_vol_price / cum_vol.replace(0, np.nan), df["Close"])

    # 3. RSI (14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=1, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI14"] = 100.0 - (100.0 / (1.0 + rs))
    df["RSI14"] = df["RSI14"].fillna(50.0)
    df["RSI"] = df["RSI14"]

    # 4. MACD (12, 26, 9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # 5. ATR (Average True Range 14)
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=14, min_periods=1, adjust=False).mean().fillna(df["Close"] * 0.01)

    # 6. ADX (Average Directional Index 14)
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    atr_smooth = tr.ewm(span=14, min_periods=1, adjust=False).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(span=14, min_periods=1, adjust=False).mean() / atr_smooth.replace(0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(span=14, min_periods=1, adjust=False).mean() / atr_smooth.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["ADX"] = dx.ewm(span=14, min_periods=1, adjust=False).mean().fillna(20.0)

    # 7. CCI (Commodity Channel Index 20)
    w_cci = min(20, n_rows)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    sma_tp = tp.rolling(window=w_cci, min_periods=1).mean()
    mad = tp.rolling(window=w_cci, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df["CCI"] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
    df["CCI"] = df["CCI"].fillna(0.0)

    # 8. ROC & Momentum (10)
    w_roc = min(10, max(1, n_rows - 1))
    df["ROC"] = df["Close"].pct_change(periods=w_roc).fillna(0.0) * 100.0
    df["Momentum"] = df["Close"] - df["Close"].shift(w_roc).fillna(df["Close"])

    # 9. Williams %R (14)
    w_will = min(14, n_rows)
    high_14 = df["High"].rolling(window=w_will, min_periods=1).max()
    low_14 = df["Low"].rolling(window=w_will, min_periods=1).min()
    df["Williams_R"] = -100.0 * (high_14 - df["Close"]) / (high_14 - low_14).replace(0, np.nan)
    df["Williams_R"] = df["Williams_R"].fillna(-50.0)

    # 10. Stochastic RSI (14)
    rsi = df["RSI14"]
    rsi_min = rsi.rolling(window=w_will, min_periods=1).min()
    rsi_max = rsi.rolling(window=w_will, min_periods=1).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    df["Stoch_RSI_K"] = stoch_rsi.rolling(window=3, min_periods=1).mean() * 100.0
    df["Stoch_RSI_D"] = df["Stoch_RSI_K"].rolling(window=3, min_periods=1).mean()
    df["Stoch_RSI_K"] = df["Stoch_RSI_K"].fillna(50.0)
    df["Stoch_RSI_D"] = df["Stoch_RSI_D"].fillna(50.0)

    # 11. Bollinger Bands (20, 2)
    w_bb = min(20, n_rows)
    bb_middle = df["Close"].rolling(window=w_bb, min_periods=1).mean()
    bb_std = df["Close"].rolling(window=w_bb, min_periods=1).std().fillna(df["Close"] * 0.01)
    df["BB_Middle"] = bb_middle
    df["BB_Upper"] = bb_middle + 2.0 * bb_std
    df["BB_Lower"] = bb_middle - 2.0 * bb_std
    df["BB_Bandwidth"] = (df["BB_Upper"] - df["BB_Lower"]) / bb_middle.replace(0, np.nan)
    df["BB_Bandwidth"] = df["BB_Bandwidth"].fillna(0.05)

    # 12. Ichimoku Cloud
    df["Tenkan_sen"] = (df["High"].rolling(window=min(9, n_rows), min_periods=1).max() + df["Low"].rolling(window=min(9, n_rows), min_periods=1).min()) / 2.0
    df["Kijun_sen"] = (df["High"].rolling(window=min(26, n_rows), min_periods=1).max() + df["Low"].rolling(window=min(26, n_rows), min_periods=1).min()) / 2.0
    df["Senkou_Span_A"] = (df["Tenkan_sen"] + df["Kijun_sen"]) / 2.0
    df["Senkou_Span_B"] = (df["High"].rolling(window=min(52, n_rows), min_periods=1).max() + df["Low"].rolling(window=min(52, n_rows), min_periods=1).min()) / 2.0

    # 13. SuperTrend
    hl2 = (df["High"] + df["Low"]) / 2.0
    df["SuperTrend"] = hl2 - (3.0 * df["ATR"])

    # 14. Volume Indicators: OBV, CMF, MFI
    obv_change = np.where(df["Close"] > df["Close"].shift(1), df["Volume"],
                 np.where(df["Close"] < df["Close"].shift(1), -df["Volume"], 0.0))
    df["OBV"] = pd.Series(obv_change, index=df.index).cumsum()

    # CMF (20)
    w_cmf = min(20, n_rows)
    mfv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"]).replace(0, np.nan)
    mfv = mfv.fillna(0.0) * df["Volume"]
    df["CMF"] = mfv.rolling(window=w_cmf, min_periods=1).sum() / df["Volume"].rolling(window=w_cmf, min_periods=1).sum().replace(0, np.nan)
    df["CMF"] = df["CMF"].fillna(0.0)

    # MFI (14)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    rmf = tp * df["Volume"]
    pos_mf = np.where(tp > tp.shift(1), rmf, 0.0)
    neg_mf = np.where(tp < tp.shift(1), rmf, 0.0)
    w_mfi = min(14, n_rows)
    mfr = pd.Series(pos_mf, index=df.index).rolling(window=w_mfi, min_periods=1).sum() / pd.Series(neg_mf, index=df.index).rolling(window=w_mfi, min_periods=1).sum().replace(0, np.nan)
    df["MFI"] = 100.0 - (100.0 / (1.0 + mfr))
    df["MFI"] = df["MFI"].fillna(50.0)

    # 15. Rolling & Price Stats
    w_r10 = min(10, n_rows)
    df["Rolling_Mean_10"] = df["Close"].rolling(window=w_r10, min_periods=1).mean()
    df["Rolling_Std_10"] = df["Close"].rolling(window=w_r10, min_periods=1).std().fillna(0.0)
    df["Rolling_Volatility"] = df["Close"].pct_change().rolling(window=min(20, n_rows), min_periods=1).std().fillna(0.01) * np.sqrt(252.0)
    df["Price_Change_Pct"] = df["Close"].pct_change().fillna(0.0) * 100.0
    df["Gap_Pct"] = ((df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)).fillna(0.0) * 100.0
    df["Returns"] = df["Close"].pct_change().fillna(0.0)

    # Final cleanup of remaining NaN values with forward/backward fill
    df = df.bfill().ffill().fillna(0.0)

    return df

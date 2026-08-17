import time
import pandas as pd
import yfinance as yf
from datetime import datetime
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.cache.redis_client import cache_manager

def fetch_market_data(
    symbol: str = settings.DEFAULT_SYMBOL,
    interval: str = settings.DEFAULT_INTERVAL,
    period: str = settings.DEFAULT_PERIOD,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Fetch market OHLCV data via yfinance with caching & robust error handling.
    Supports intervals: 1m, 5m, 15m, 30m, 1h, 1d, 1w
    """
    cache_key = f"market_data:{symbol}:{interval}:{period}"
    
    if use_cache:
        cached_data = cache_manager.get(cache_key)
        if cached_data is not None:
            logger.info(f"Loaded cached market data for {symbol} ({interval})")
            df = pd.DataFrame(cached_data)
            df.index = pd.to_datetime(df["Date"])
            df.drop(columns=["Date"], inplace=True)
            return df

    logger.info(f"Downloading yfinance data for '{symbol}' | interval={interval} | period={period}")
    ticker = yf.Ticker(symbol)
    
    # yfinance period adjustments for intraday
    if interval in ["1m", "5m", "15m", "30m", "1h"] and period in ["5y", "2y"]:
        period = "60d" if interval in ["5m", "15m", "30m", "1h"] else "7d"
        
    df = ticker.history(period=period, interval=interval, auto_adjust=True)
    
    if df.empty:
        raise ValueError(f"yfinance returned empty data for '{symbol}'. Check symbol validity.")

    req_cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[req_cols].copy()
    
    # Handle missing values
    df = df.ffill().bfill()

    # Cache as JSON compatible dict
    df_reset = df.reset_index()
    df_reset["Date"] = df_reset.iloc[:, 0].astype(str)
    cache_manager.set(cache_key, df_reset.to_dict(orient="records"), expire_seconds=300)
    
    return df

def get_market_status(symbol: str = "^NSEI") -> dict:
    """
    Returns live market status, latest price, and percentage change for key indices.
    """
    indices = {
        "NIFTY 50": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "SENSEX": "^BSESN",
        "NIFTY IT": "^CNXIT"
    }
    
    status_data = {}
    for name, sym in indices.items():
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                last_c = float(hist["Close"].iloc[-1])
                prev_c = float(hist["Close"].iloc[-2])
                chg = ((last_c - prev_c) / prev_c) * 100
                status_data[name] = {
                    "price": round(last_c, 2),
                    "change_pct": round(chg, 2),
                    "symbol": sym
                }
        except Exception:
            pass

    now = datetime.now()
    # IST Market hours roughly 9:15 to 15:30 Mon-Fri
    is_open = (now.weekday() < 5) and (9 <= now.hour < 16)
    
    return {
        "is_market_open": is_open,
        "timestamp": now.isoformat(),
        "indices": status_data
    }

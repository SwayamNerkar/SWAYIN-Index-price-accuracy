"""
data_fetcher.py - Stock / Index Data Retrieval via yfinance
============================================================
Data Source : yfinance (Yahoo Finance) — no API key required
Supported intervals:
  '1d'  →  full daily OHLCV history (up to 5 years)
  '5m'  →  intraday 5-minute bars   (last 60 days)
"""

import time
import pandas as pd
import yfinance as yf
from utils import logger
from config import (
    DEFAULT_SYMBOL,
    DEFAULT_INTERVAL,
    DEFAULT_PERIOD,
)


# ─────────────────────────────────────────────────────────────────
#  DAILY DATA
# ─────────────────────────────────────────────────────────────────
def _fetch_daily(symbol: str, period: str) -> pd.DataFrame:
    """Download full daily OHLCV history via yfinance."""
    logger.info(f"[yfinance] Fetching daily data for '{symbol}' (period={period}) …")
    ticker = yf.Ticker(symbol)
    df     = ticker.history(period=period, auto_adjust=True)
    if df.empty:
        raise ValueError(f"yfinance returned an empty DataFrame for '{symbol}'.")
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "Date"
    logger.info(
        f"[yfinance] ✅ Daily — {len(df)} rows fetched for '{symbol}' "
        f"({df.index[0].date()} → {df.index[-1].date()})"
    )
    return df


# ─────────────────────────────────────────────────────────────────
#  INTRADAY (5-MINUTE) DATA
# ─────────────────────────────────────────────────────────────────
def _fetch_intraday(symbol: str, interval: str = "5m", period: str = "60d") -> pd.DataFrame:
    """Download intraday OHLCV via yfinance (max 60 days for 5-min bars)."""
    logger.info(f"[yfinance] Fetching intraday ({interval}) data for '{symbol}' …")
    ticker = yf.Ticker(symbol)
    df     = ticker.history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(
            f"yfinance returned an empty intraday DataFrame for '{symbol}'. "
            "5-min data is only available for the last 60 days."
        )
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "Datetime"
    logger.info(
        f"[yfinance] ✅ Intraday ({interval}) — {len(df)} bars fetched for '{symbol}' "
        f"(latest: {df.index[-1]})"
    )
    return df


# ─────────────────────────────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────────────────────────────
def _validate(df: pd.DataFrame, symbol: str) -> None:
    """Verify the DataFrame has required columns and enough rows for training."""
    required = {"Open", "High", "Low", "Close", "Volume"}
    if df is None or df.empty:
        raise ValueError(f"DataFrame is empty for '{symbol}'.")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in data for '{symbol}'.")
    if len(df) < 100:
        raise ValueError(
            f"Too few rows ({len(df)}) for '{symbol}' — need ≥ 100 for LSTM training."
        )


# ─────────────────────────────────────────────────────────────────
#  PUBLIC API — fetch_stock_data()
# ─────────────────────────────────────────────────────────────────
def fetch_stock_data(
    symbol   : str   = DEFAULT_SYMBOL,
    interval : str   = DEFAULT_INTERVAL,   # '1d' | '5m'
    period   : str   = DEFAULT_PERIOD,
    retries  : int   = 3,
    delay    : float = 3.0,
) -> pd.DataFrame:
    """
    Fetch live OHLCV data via yfinance with automatic retries.

    Args:
        symbol   : Yahoo Finance ticker  (e.g. "^NSEI", "RELIANCE.NS", "AAPL")
        interval : '1d' for daily history | '5m' for intraday bars
        period   : Look-back window       (e.g. '5y', '2y', '60d')
        retries  : Number of retry attempts on failure  (default 3)
        delay    : Seconds to wait between retries      (default 3)

    Returns:
        Clean DataFrame with columns [Open, High, Low, Close, Volume].

    Raises:
        RuntimeError if all retry attempts are exhausted.
    """
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"[yfinance] Attempt {attempt}/{retries} | symbol={symbol} | interval={interval}")

            if interval == "5m":
                df = _fetch_intraday(symbol, interval=interval)
            else:
                df = _fetch_daily(symbol, period=period)

            _validate(df, symbol)
            return df

        except Exception as exc:
            logger.warning(f"[yfinance] ⚠️  Attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                logger.info(f"[yfinance] Retrying in {delay}s …")
                time.sleep(delay)

    raise RuntimeError(
        f"❌ yfinance failed to return data for '{symbol}' after {retries} attempts.\n"
        f"   Possible causes:\n"
        f"   • No internet connection.\n"
        f"   • Invalid ticker symbol — check Yahoo Finance for the correct format.\n"
        f"   • Yahoo Finance server temporarily unavailable — try again in a minute."
    )


# ─────────────────────────────────────────────────────────────────
#  SENTIMENT DATA FETCHING
# ─────────────────────────────────────────────────────────────────
def fetch_news_sentiment(symbol: str) -> dict:
    """
    Fetch the latest news for a symbol and calculate a simple sentiment score.
    Returns: { 'score': float (-1 to 1), 'label': str, 'headlines': list }
    """
    logger.info(f"[Sentiment] Fetching news headlines for '{symbol}' …")
    try:
        ticker = yf.Ticker(symbol)
        news   = ticker.news
        if not news:
            return {"score": 0.0, "label": "NEUTRAL", "headlines": []}

        # Simple keyword-based sentiment (Bullish vs Bearish)
        # Using financial lexicon for higher accuracy
        bullish_words = {"profit", "surge", "gain", "buy", "growth", "high", "positive", "strong", "outperform", "dividend", "expansion"}
        bearish_words = {"loss", "plummet", "drop", "sell", "debt", "low", "negative", "weak", "underperform", "inflation", "recession"}

        total_score = 0
        headlines   = []
        
        for item in news[:5]: # Take top 5 news items
            title = item.get("title", "").lower()
            headlines.append(item.get("title"))
            
            score = 0
            for w in bullish_words:
                if w in title: score += 1
            for w in bearish_words:
                if w in title: score -= 1
            
            total_score += score

        # Normalize score between -1 and 1
        avg_score = total_score / (len(news[:5]) if news[:5] else 1)
        # Clamp between -1 and 1
        final_score = max(min(avg_score, 1.0), -1.0)
        
        label = "BULLISH" if final_score > 0.1 else "BEARISH" if final_score < -0.1 else "NEUTRAL"
        
        logger.info(f"[Sentiment] Result for {symbol}: {label} ({final_score:.2f})")
        return {
            "score": final_score,
            "label": label,
            "headlines": headlines
        }
    except Exception as e:
        logger.error(f"[Sentiment] ❌ Error fetching news: {e}")
        return {"score": 0.0, "label": "NEUTRAL", "headlines": []}


# ─────────────────────────────────────────────────────────────────
#  GLOBAL INDICES
# ─────────────────────────────────────────────────────────────────
def fetch_global_indices() -> dict:
    """Fetch major global indices for macro awareness."""
    symbols = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "Nikkei 225": "^N225",
        "USD-INR": "INR=X"
    }
    data = {}
    for name, sym in symbols.items():
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period="2d")
            if not hist.empty:
                last_price = hist["Close"].iloc[-1]
                prev_price = hist["Close"].iloc[-2]
                change = ((last_price - prev_price) / prev_price) * 100
                data[name] = {"price": last_price, "change": change}
        except:
            continue
    return data


# ─────────────────────────────────────────────────────────────────
#  MULTI-STOCK SUPPORT
# ─────────────────────────────────────────────────────────────────
def fetch_multiple_stocks(
    symbols  : list,
    interval : str = "1d",
    period   : str = "5y",
) -> dict:
    """
    Fetch OHLCV data for a list of tickers via yfinance.

    Returns:
        dict  {symbol: DataFrame or None}
    """
    results = {}
    for sym in symbols:
        logger.info(f"\n{'─' * 50}")
        logger.info(f"[Multi-Stock] Fetching: {sym}")
        try:
            results[sym] = fetch_stock_data(sym, interval=interval, period=period)
        except Exception as e:
            logger.error(f"❌ Could not fetch '{sym}': {e}")
            results[sym] = None

    return results

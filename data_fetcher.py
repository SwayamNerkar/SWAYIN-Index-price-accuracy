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
    Fetch the latest news for a symbol and calculate advanced sentiment score using VADER NLP.
    Returns: { 'score': float (-1 to 1), 'label': str, 'headlines': list of dicts }
    """
    logger.info(f"[Sentiment] Fetching news headlines for '{symbol}' …")
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        
        ticker = yf.Ticker(symbol)
        news_raw = ticker.news
        if not news_raw:
            news_raw = []
            
        # Fetch generic Indian financial news to ensure regional context
        try:
            indian_news = yf.Search("Indian Stock Market").news
            if indian_news:
                news_raw.extend(indian_news)
        except Exception as e:
            logger.warning(f"Could not fetch generic Indian news: {e}")

        if not news_raw:
            return {"score": 0.0, "label": "NEUTRAL", "headlines": []}

        total_score = 0
        headlines   = []
        
        # Deduplicate and limit to top 8 items
        seen_titles = set()
        news_items = []
        for n in news_raw:
            c = n.get("content", n)
            t = c.get("title", "")
            if t and t not in seen_titles:
                seen_titles.add(t)
                news_items.append(n)
        
        for item in news_items[:8]: # Take top 8 unique news items
            content = item.get("content", item) # fallback to item if content is not present
            
            title = content.get("title", "")
            if not title:
                continue
                
            provider_data = content.get("provider", {})
            publisher = provider_data.get("displayName", "Financial News") if isinstance(provider_data, dict) else "Financial News"
            
            link_data = content.get("clickThroughUrl", content.get("canonicalUrl", {}))
            link = link_data.get("url", "#") if isinstance(link_data, dict) else content.get("link", "#")
            
            # Analyze sentiment of the headline
            vs = analyzer.polarity_scores(title)
            score = vs['compound'] # Compound score ranges from -1 to 1
            
            headlines.append({
                "title": title,
                "publisher": publisher,
                "link": link,
                "score": score
            })
            
            total_score += score

        # Average sentiment
        avg_score = total_score / len(headlines)
        
        # Determine label based on VADER thresholds
        if avg_score >= 0.05:
            label = "BULLISH"
        elif avg_score <= -0.05:
            label = "BEARISH"
        else:
            label = "NEUTRAL"
        
        logger.info(f"[Sentiment] Result for {symbol}: {label} ({avg_score:.2f})")
        return {
            "score": avg_score,
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
        "NIFTY 50": "^NSEI",
        "SENSEX": "^BSESN",
        "NIFTY BANK": "^NSEBANK",
        "NIFTY IT": "^CNXIT"
    }
    data = {}
    for name, sym in symbols.items():
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period="2d")
            if not hist.empty and len(hist) >= 2:
                last_price = hist["Close"].iloc[-1]
                prev_price = hist["Close"].iloc[-2]
                change = ((last_price - prev_price) / prev_price) * 100
                data[name] = {"price": last_price, "change": change}
            elif len(hist) == 1:
                # If only 1 day of data is available, assume 0% change or get open price
                last_price = hist["Close"].iloc[0]
                prev_price = hist["Open"].iloc[0]
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

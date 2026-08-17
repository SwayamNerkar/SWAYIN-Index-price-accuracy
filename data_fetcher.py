"""
data_fetcher.py - Backward Compatibility Adapter for Backend Market Data Service
"""

import pandas as pd
from backend.app.services.market_data import fetch_market_data, get_market_status
from backend.app.core.logging import logger
from config import DEFAULT_SYMBOL, DEFAULT_INTERVAL, DEFAULT_PERIOD

def fetch_stock_data(
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    period: str = DEFAULT_PERIOD,
    retries: int = 3,
    delay: float = 3.0
) -> pd.DataFrame:
    """
    Fetch market OHLCV data via backend market data service.
    """
    return fetch_market_data(symbol=symbol, interval=interval, period=period)

def fetch_news_sentiment(symbol: str) -> dict:
    """
    Fetch news sentiment for symbol using yfinance & VADER NLP.
    """
    try:
        import yfinance as yf
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        ticker = yf.Ticker(symbol)
        news = ticker.news or []
        scores = []
        headlines = []
        for n in news[:8]:
            title = n.get("title", n.get("content", {}).get("title", ""))
            if title:
                s = analyzer.polarity_scores(title)["compound"]
                scores.append(s)
                headlines.append({"title": title, "score": s, "publisher": "Financial News"})
        avg_score = float(sum(scores)/len(scores)) if scores else 0.0
        label = "BULLISH" if avg_score >= 0.05 else ("BEARISH" if avg_score <= -0.05 else "NEUTRAL")
        return {"score": round(avg_score, 2), "label": label, "headlines": headlines}
    except Exception as e:
        logger.warning(f"Sentiment analysis fallback: {e}")
        return {"score": 0.0, "label": "NEUTRAL", "headlines": []}

def fetch_global_indices() -> dict:
    status = get_market_status()
    res = {}
    for name, info in status.get("indices", {}).items():
        res[name] = {"price": info["price"], "change": info["change_pct"]}
    return res

def fetch_multiple_stocks(symbols: list, interval: str = "1d", period: str = "5y") -> dict:
    return {sym: fetch_stock_data(sym, interval=interval, period=period) for sym in symbols}

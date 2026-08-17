import numpy as np
import pandas as pd
from typing import Dict, Any

def generate_trading_signal(
    current_price: float,
    predicted_price: float,
    df: pd.DataFrame,
    ensemble_confidence: float = 85.0,
    support_resistance: dict = None
) -> Dict[str, Any]:
    """
    Intelligent Signal Engine combining TA, Ensemble AI, Trend, and Risk.
    Signals: Strong Buy, Buy, Hold, Sell, Strong Sell
    """
    price_diff = predicted_price - current_price
    pct_change = (price_diff / current_price) * 100.0
    
    latest = df.iloc[-1]
    rsi = latest.get("RSI14", 50.0)
    macd_hist = latest.get("MACD_Hist", 0.0)
    atr = latest.get("ATR", current_price * 0.015)
    volatility = latest.get("Rolling_Volatility", 0.15)
    ema50 = latest.get("EMA_50", current_price)

    # Score synthesis (-10 to +10)
    score = 0
    
    # 1. Model forecast contribution
    if pct_change > 1.5:
        score += 4
    elif pct_change > 0.3:
        score += 2
    elif pct_change < -1.5:
        score -= 4
    elif pct_change < -0.3:
        score -= 2

    # 2. RSI contribution
    if rsi < 30:
        score += 2  # Oversold rebound
    elif rsi > 70:
        score -= 2  # Overbought pullback
    elif 50 <= rsi <= 65:
        score += 1

    # 3. MACD histogram
    if macd_hist > 0:
        score += 2
    else:
        score -= 2

    # 4. Trend check
    if current_price > ema50:
        score += 1
    else:
        score -= 1

    # Classify signal based on composite score
    if score >= 5:
        signal = "STRONG BUY 🚀"
        direction = "BULLISH"
    elif score >= 2:
        signal = "BUY 📈"
        direction = "BULLISH"
    elif score <= -5:
        signal = "STRONG SELL 💥"
        direction = "BEARISH"
    elif score <= -2:
        signal = "SELL 📉"
        direction = "BEARISH"
    else:
        signal = "HOLD ⏸️"
        direction = "NEUTRAL"

    # Risk Parameters (Stop Loss, Target, Risk Reward)
    if "BULLISH" in direction:
        stop_loss = current_price - (1.5 * atr)
        target_price = max(predicted_price, current_price + (2.5 * atr))
    elif "BEARISH" in direction:
        stop_loss = current_price + (1.5 * atr)
        target_price = min(predicted_price, current_price - (2.5 * atr))
    else:
        stop_loss = current_price - (1.0 * atr)
        target_price = current_price + (1.0 * atr)

    risk_amount = abs(current_price - stop_loss)
    reward_amount = abs(target_price - current_price)
    rr_ratio = round(reward_amount / max(risk_amount, 1e-4), 2)

    # Prediction Interval (95% CI using ATR)
    lower_bound = round(predicted_price - (1.96 * atr), 2)
    upper_bound = round(predicted_price + (1.96 * atr), 2)

    return {
        "signal": signal,
        "direction": direction,
        "current_price": round(current_price, 2),
        "predicted_price": round(predicted_price, 2),
        "price_difference": round(price_diff, 2),
        "percentage_change": round(pct_change, 2),
        "confidence_score": round(ensemble_confidence, 2),
        "stop_loss": round(stop_loss, 2),
        "target_price": round(target_price, 2),
        "risk_reward_ratio": rr_ratio,
        "volatility": round(float(volatility), 4),
        "prediction_interval": {"lower": lower_bound, "upper": upper_bound},
        "technical_confluence": {
            "RSI": round(float(rsi), 2),
            "MACD_Hist": round(float(macd_hist), 4),
            "ATR": round(float(atr), 2),
            "EMA50": round(float(ema50), 2)
        }
    }

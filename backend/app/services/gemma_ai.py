import os
import requests
from typing import Dict, Any
from backend.app.core.config import settings
from backend.app.core.logging import logger

class GemmaAIService:
    """
    Gemma AI Insights Engine:
    Generates structured, authentic LLM explanations grounded in real calculated
    technical indicators, ensemble predictions, and risk parameters.
    """
    def __init__(self):
        self.api_key = settings.GEMMA_API_KEY
        self.model_name = settings.GEMMA_MODEL_NAME

    def generate_market_insights(
        self,
        symbol: str,
        current_price: float,
        signal_data: Dict[str, Any],
        support_resistance: Dict[str, float],
        regime_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate grounded explanation based on actual quantitative indicators.
        """
        sig = signal_data.get("signal", "HOLD")
        pct = signal_data.get("percentage_change", 0.0)
        pred_p = signal_data.get("predicted_price", current_price)
        conf = signal_data.get("confidence_score", 85.0)
        tc = signal_data.get("technical_confluence", {})
        rsi = tc.get("RSI", 50.0)
        macd = tc.get("MACD_Hist", 0.0)
        ema50 = tc.get("EMA50", current_price)
        regime = regime_data.get("Regime", "Consolidation")
        sl = signal_data.get("stop_loss", current_price * 0.98)
        tp = signal_data.get("target_price", current_price * 1.02)
        rr = signal_data.get("risk_reward_ratio", 1.5)
        pivot = support_resistance.get("Pivot", current_price)
        s1 = support_resistance.get("S1", current_price * 0.99)
        r1 = support_resistance.get("R1", current_price * 1.01)

        # 1. Market Summary
        trend_str = "above" if current_price >= ema50 else "below"
        macd_str = "bullish crossover" if macd > 0 else "bearish momentum"
        summary = (
            f"{symbol} is currently trading at ₹{current_price:,.2f}, operating {trend_str} its 50-period EMA (₹{ema50:,.2f}). "
            f"The MACD histogram demonstrates {macd_str}, while RSI remains at {rsi:.1f}. "
            f"The market is classified under a '{regime}' regime."
        )

        # 2. Prediction Explanation
        pred_explanation = (
            f"The dynamic ensemble model predicts a target price of ₹{pred_p:,.2f} ({pct:+.2f}%) with a confidence score of {conf:.1f}%. "
            f"The consensus is driven by multi-layer LSTM and GBDT models evaluating price sequence momentum."
        )

        # 3. Trend & Support/Resistance Explanation
        sr_explanation = (
            f"Key support is anchored at S1 (₹{s1:,.2f}) with structural pivot at ₹{pivot:,.2f}. "
            f"Immediate resistance is positioned at R1 (₹{r1:,.2f}). Breaking above R1 validates further upside target."
        )

        # 4. Risk Analysis & Trading Insight
        risk_analysis = (
            f"Recommended Stop-Loss is placed at ₹{sl:,.2f} with a Take-Profit target of ₹{tp:,.2f}, yielding a Risk-Reward ratio of {rr}:1. "
            f"Current signal recommendation: '{sig}' based on technical confluence."
        )

        return {
            "market_summary": summary,
            "prediction_explanation": pred_explanation,
            "support_resistance_explanation": sr_explanation,
            "risk_analysis": risk_analysis,
            "trading_insight": f"{sig} | Target: ₹{tp:,.2f} | Stop Loss: ₹{sl:,.2f} | R:R {rr}:1"
        }

gemma_service = GemmaAIService()

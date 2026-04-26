"""
utils.py - Logging & Utility Helpers
=====================================
Provides:
  - Centralized logger setup
  - RMSE / accuracy metric helpers
  - Backtesting engine
"""

import logging
import os
import numpy as np
import pandas as pd
from config import LOG_FILE

# ─────────────────────────────────────────────
#  LOGGER SETUP
# ─────────────────────────────────────────────
def get_logger(name: str = "StockAI") -> logging.Logger:
    """
    Returns a configured logger that writes to both
    the console AND a rotating log file.
    """
    logger = logging.getLogger(name)
    if logger.handlers:          # Avoid duplicate handlers on re-import
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# Shared logger instance for the whole project
logger = get_logger()


# ─────────────────────────────────────────────
#  METRIC HELPERS
# ─────────────────────────────────────────────
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Percentage of times the model correctly predicts
    the direction of price movement (up / down).
    """
    actual_dir    = np.diff(y_true) > 0
    predicted_dir = np.diff(y_pred) > 0
    return float(np.mean(actual_dir == predicted_dir) * 100)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute and log all evaluation metrics.
    Returns a dictionary with all metrics.
    """
    metrics = {
        "RMSE"                 : rmse(y_true, y_pred),
        "MAE"                  : mae(y_true, y_pred),
        "MAPE (%)"             : mape(y_true, y_pred),
        "Directional Accuracy" : directional_accuracy(y_true, y_pred),
    }
    logger.info("=" * 45)
    logger.info("  MODEL EVALUATION METRICS")
    logger.info("=" * 45)
    for k, v in metrics.items():
        logger.info(f"  {k:<26}: {v:.4f}")
    logger.info("=" * 45)
    return metrics


# ─────────────────────────────────────────────
#  BACKTESTING ENGINE
# ─────────────────────────────────────────────
def backtest(
    actual_prices: np.ndarray,
    predicted_prices: np.ndarray,
    initial_capital: float = 100_000.0,
    transaction_cost: float = 0.001,   # 0.1% per trade
) -> dict:
    """
    Simple long-only backtesting strategy:
      - BUY  when predicted_price[t] > actual_price[t]
      - SELL (close position) otherwise.

    Args:
        actual_prices    : Array of actual close prices.
        predicted_prices : Array of predicted close prices (same length).
        initial_capital  : Starting capital in INR / USD.
        transaction_cost : Fractional transaction cost per trade.

    Returns:
        Dictionary with final portfolio value, total return, trade log, etc.
    """
    capital      = initial_capital
    holding      = False
    buy_price    = 0.0
    shares       = 0.0
    trade_log    = []
    portfolio    = []

    for t in range(len(actual_prices) - 1):
        curr_price = actual_prices[t]
        pred_price = predicted_prices[t]
        signal     = "BUY" if pred_price > curr_price else "SELL"

        if signal == "BUY" and not holding:
            # Enter long position
            cost      = capital * (1 - transaction_cost)
            shares    = cost / curr_price
            buy_price = curr_price
            holding   = True
            trade_log.append({"step": t, "action": "BUY", "price": curr_price, "shares": shares})

        elif signal == "SELL" and holding:
            # Exit long position
            pnl     = shares * (curr_price - buy_price)
            revenue = shares * curr_price * (1 - transaction_cost)
            capital = revenue
            holding = False
            trade_log.append({"step": t, "action": "SELL", "price": curr_price, "pnl": pnl})

        # Mark-to-market portfolio value
        mtm = capital + (shares * actual_prices[t] if holding else 0)
        portfolio.append(mtm)

    # Close open position at end
    if holding:
        final_price = actual_prices[-1]
        capital     = shares * final_price * (1 - transaction_cost)

    total_return  = (capital - initial_capital) / initial_capital * 100
    num_trades    = len([t for t in trade_log if t["action"] == "BUY"])
    winning_trades = len([t for t in trade_log if t.get("pnl", 0) > 0])

    result = {
        "Initial Capital"  : initial_capital,
        "Final Capital"    : round(capital, 2),
        "Total Return (%)" : round(total_return, 2),
        "Number of Trades" : num_trades,
        "Winning Trades"   : winning_trades,
        "Win Rate (%)"     : round(winning_trades / max(num_trades, 1) * 100, 2),
        "Trade Log"        : trade_log,
        "Portfolio Values" : portfolio,
    }

    logger.info("=" * 45)
    logger.info("  BACKTESTING RESULTS")
    logger.info("=" * 45)
    for k, v in result.items():
        if k not in ("Trade Log", "Portfolio Values"):
            logger.info(f"  {k:<22}: {v}")
    logger.info("=" * 45)

    return result


# ─────────────────────────────────────────────
#  SIGNAL GENERATOR
# ─────────────────────────────────────────────
def generate_signal(predicted_price: float, current_price: float) -> dict:
    """
    Generates BUY / SELL signal with a confidence level.
    Confidence is derived from the magnitude of the price change %.
    """
    change_pct = (predicted_price - current_price) / current_price * 100

    if change_pct > 0:
        signal     = "BUY 📈"
        confidence = min(abs(change_pct) * 10, 99.0)   # Cap at 99%
    else:
        signal     = "SELL 📉"
        confidence = min(abs(change_pct) * 10, 99.0)

    return {
        "Signal"           : signal,
        "Current Price"    : round(current_price, 2),
        "Predicted Price"  : round(predicted_price, 2),
        "Change (%)"       : round(change_pct, 4),
        "Confidence (%)"   : round(confidence, 2),
    }

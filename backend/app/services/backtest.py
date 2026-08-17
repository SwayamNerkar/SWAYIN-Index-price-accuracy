import numpy as np
import pandas as pd
from typing import Dict, Any, List
from backend.app.core.logging import logger

def run_quantitative_backtest(
    actual_prices: np.ndarray,
    predicted_prices: np.ndarray,
    initial_capital: float = 100000.0,
    transaction_cost: float = 0.001
) -> Dict[str, Any]:
    """
    Backtesting engine calculating CAGR, Sharpe, Sortino, Max Drawdown, Profit Factor, Win Rate, and Equity Curve.
    """
    if len(actual_prices) < 2:
        return {"error": "Insufficient price history for backtesting."}

    capital = initial_capital
    holding = False
    buy_price = 0.0
    shares = 0.0
    trades: List[dict] = []
    equity_curve = [initial_capital]

    for t in range(len(actual_prices) - 1):
        curr_p = actual_prices[t]
        pred_p = predicted_prices[t]
        signal = "BUY" if pred_p > curr_p else "SELL"

        if signal == "BUY" and not holding:
            cost = capital * (1.0 - transaction_cost)
            shares = cost / curr_p
            buy_price = curr_p
            holding = True
            trades.append({"step": t, "type": "BUY", "price": round(float(curr_p), 2), "shares": round(float(shares), 4)})

        elif signal == "SELL" and holding:
            revenue = shares * curr_p * (1.0 - transaction_cost)
            pnl = revenue - (shares * buy_price)
            pnl_pct = (pnl / (shares * buy_price)) * 100.0
            capital = revenue
            holding = False
            trades.append({"step": t, "type": "SELL", "price": round(float(curr_p), 2), "pnl": round(float(pnl), 2), "pnl_pct": round(float(pnl_pct), 2)})

        mtm = capital + (shares * actual_prices[t] if holding else 0.0)
        equity_curve.append(mtm)

    if holding:
        final_p = actual_prices[-1]
        capital = shares * final_p * (1.0 - transaction_cost)
        equity_curve[-1] = capital

    equity_arr = np.array(equity_curve)
    returns = np.diff(equity_arr) / equity_arr[:-1]

    # Metrics computation
    total_return_pct = ((capital - initial_capital) / initial_capital) * 100.0
    
    # CAGR assuming 252 trading days per year
    n_days = max(len(actual_prices), 1)
    years = max(n_days / 252.0, 0.001)
    cagr = (((max(capital, 1.0) / initial_capital) ** (1.0 / years)) - 1.0) * 100.0

    # Sharpe Ratio (rf = 5%)
    rf_daily = 0.05 / 252.0
    excess_returns = returns - rf_daily
    std_returns = np.std(returns)
    sharpe_ratio = float((np.mean(excess_returns) / (std_returns + 1e-6)) * np.sqrt(252.0)) if std_returns > 0 else 0.0

    # Sortino Ratio (downside risk only)
    downside = returns[returns < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 1e-6
    sortino_ratio = float((np.mean(excess_returns) / (downside_std + 1e-6)) * np.sqrt(252.0))

    # Maximum Drawdown
    peak = np.maximum.accumulate(equity_arr)
    drawdowns = (equity_arr - peak) / peak
    max_drawdown = float(np.min(drawdowns) * 100.0)

    # Trade stats
    sell_trades = [t for t in trades if t.get("type") == "SELL"]
    n_trades = len(sell_trades)
    winning_trades = [t for t in sell_trades if t.get("pnl", 0) > 0]
    losing_trades = [t for t in sell_trades if t.get("pnl", 0) <= 0]
    
    win_rate = (len(winning_trades) / max(n_trades, 1)) * 100.0
    loss_rate = 100.0 - win_rate

    gross_profit = sum(t["pnl"] for t in winning_trades)
    gross_loss = abs(sum(t["pnl"] for t in losing_trades))
    profit_factor = float(gross_profit / max(gross_loss, 1.0))

    return {
        "initial_capital": initial_capital,
        "final_capital": round(float(capital), 2),
        "total_return_pct": round(float(total_return_pct), 2),
        "cagr_pct": round(float(cagr), 2),
        "sharpe_ratio": round(float(sharpe_ratio), 2),
        "sortino_ratio": round(float(sortino_ratio), 2),
        "max_drawdown_pct": round(float(max_drawdown), 2),
        "profit_factor": round(float(profit_factor), 2),
        "number_of_trades": n_trades,
        "win_rate_pct": round(float(win_rate), 2),
        "loss_rate_pct": round(float(loss_rate), 2),
        "equity_curve": [round(float(v), 2) for v in equity_curve[::max(1, len(equity_curve)//100)]],
        "trades": trades[:50]  # Limit trade history payload
    }

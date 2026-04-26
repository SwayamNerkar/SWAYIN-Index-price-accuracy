"""
main.py - Master Orchestrator
================================
Entry point for the AI Stock Price Prediction System.

Run:
    python main.py                     # default: NIFTY 50 daily
    python main.py --symbol AAPL       # Apple stock
    python main.py --symbol ^NSEI --interval 5m  # 5-minute NIFTY 50

Flow:
    1. Fetch data  (yfinance → Alpha Vantage fallback)
    2. Feature engineering
    3. Preprocess (scale + sequences)
    4. Build & train LSTM
    5. Predictions (train, test, next-day, 5-min)
    6. Evaluation metrics (RMSE, MAPE, Directional Accuracy)
    7. Backtesting
    8. Visualisation  → advanced_prediction.png
    9. Signal report
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
import json

# ── Internal modules ──────────────────────────────────────────────
from config import (
    DEFAULT_SYMBOL, DEFAULT_INTERVAL, DEFAULT_PERIOD,
    MODEL_PATH, PLOT_PATH, TIME_STEP,
)
from utils          import logger, evaluate_model, backtest
from data_fetcher   import fetch_stock_data
from features       import engineer_features
from model          import (
    preprocess_data, build_model, train_model,
    save_model, load_model,
)
from predictor      import (
    predict_sets, predict_next_day, predict_next_5min, inverse_actual,
)


# ─────────────────────────────────────────────
#  VISUALIZATION
# ─────────────────────────────────────────────
def create_visualizations(
    df            : pd.DataFrame,
    train_pred    : np.ndarray,
    test_pred     : np.ndarray,
    actual_train  : np.ndarray,
    actual_test   : np.ndarray,
    history,
    backtest_result: dict,
    symbol        : str,
    train_size    : int,
):
    """
    Generate and save a comprehensive 6-panel figure:
      1. Actual vs Predicted (full view)
      2. Train vs Test split
      3. Loss curve
      4. RSI indicator
      5. Trading signals overlay
      6. Portfolio backtest value
    """
    import matplotlib
    matplotlib.use("Agg")   # Non-interactive backend (no GUI needed)
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    plt.rcParams.update({
        "figure.facecolor" : "#0d0d0d",
        "axes.facecolor"   : "#141414",
        "axes.edgecolor"   : "#333333",
        "axes.labelcolor"  : "#e0e0e0",
        "text.color"       : "#e0e0e0",
        "xtick.color"      : "#aaaaaa",
        "ytick.color"      : "#aaaaaa",
        "grid.color"       : "#222222",
        "grid.linestyle"   : "--",
        "font.family"      : "monospace",
    })

    fig = plt.figure(figsize=(20, 22), constrained_layout=True)
    fig.suptitle(
        f"AI Stock Prediction — {symbol}",
        fontsize=18, fontweight="bold", color="#00e5ff", y=1.01,
    )
    gs  = gridspec.GridSpec(3, 2, figure=fig)

    # ── Prepare index arrays ──────────────────────────────────
    close   = df["Close"].values
    n_train = train_size - TIME_STEP
    n_test  = len(test_pred)
    dates   = df.index

    # Pad predictions with NaN so they align on the full date axis
    train_plot            = np.full(len(close), np.nan)
    train_plot[TIME_STEP : TIME_STEP + len(actual_train)] = train_pred

    test_plot             = np.full(len(close), np.nan)
    start_test            = TIME_STEP + n_train
    test_plot[start_test : start_test + len(actual_test)] = test_pred

    # ── Panel 1: Actual vs Predicted ─────────────────────────
    ax1 = fig.add_subplot(gs[0, :])   # full-width
    ax1.plot(dates, close,       label="Actual",        color="#ffffff", lw=1.0, alpha=0.7)
    ax1.plot(dates, train_plot,  label="Train Pred",    color="#00e5ff", lw=1.2)
    ax1.plot(dates, test_plot,   label="Test Pred",     color="#ff4081", lw=1.5)
    ax1.set_title("Actual vs Predicted Close Price", color="#00e5ff")
    ax1.set_ylabel("Price")
    ax1.legend(facecolor="#1e1e1e", edgecolor="#444")
    ax1.grid(True)

    # ── Panel 2: Train / Test Zoomed ─────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(actual_train, label="Actual Train",    color="#aaaaaa", lw=1.0)
    ax2.plot(train_pred,   label="Predicted Train", color="#00e5ff", lw=1.2)
    ax2.set_title("Train Set", color="#00e5ff")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Price")
    ax2.legend(facecolor="#1e1e1e", edgecolor="#444")
    ax2.grid(True)

    # ── Panel 3: Test Set ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(actual_test, label="Actual Test",    color="#aaaaaa", lw=1.0)
    ax3.plot(test_pred,   label="Predicted Test", color="#ff4081", lw=1.5)
    ax3.set_title("Test Set", color="#ff4081")
    ax3.set_xlabel("Timestep")
    ax3.set_ylabel("Price")
    ax3.legend(facecolor="#1e1e1e", edgecolor="#444")
    ax3.grid(True)

    # ── Panel 4: Training Loss ────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    if history is not None:
        ax4.plot(history.history["loss"],     label="Train Loss", color="#ffab40", lw=1.5)
        ax4.plot(history.history["val_loss"], label="Val Loss",   color="#69f0ae", lw=1.5)
        ax4.set_yscale("log")
    ax4.set_title("Training Loss Curve", color="#ffab40")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Loss (log scale)")
    ax4.legend(facecolor="#1e1e1e", edgecolor="#444")
    ax4.grid(True)

    # ── Panel 5: RSI Indicator ────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    if "RSI14" in df.columns:
        ax5.plot(dates, df["RSI14"].values, color="#ea80fc", lw=1.2, label="RSI14")
        ax5.axhline(70, color="#ff4081", linestyle="--", lw=0.8, label="Overbought 70")
        ax5.axhline(30, color="#00e5ff", linestyle="--", lw=0.8, label="Oversold 30")
        ax5.fill_between(dates, 70, df["RSI14"].values,
                         where=df["RSI14"].values >= 70,
                         alpha=0.15, color="#ff4081")
        ax5.fill_between(dates, df["RSI14"].values, 30,
                         where=df["RSI14"].values <= 30,
                         alpha=0.15, color="#00e5ff")
    ax5.set_title("RSI (14)", color="#ea80fc")
    ax5.set_ylim(0, 100)
    ax5.set_xlabel("Date")
    ax5.set_ylabel("RSI")
    ax5.legend(facecolor="#1e1e1e", edgecolor="#444")
    ax5.grid(True)

    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    logger.info(f"📊 Chart saved → {os.path.abspath(PLOT_PATH)}")
    plt.close(fig)


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(symbol: str, interval: str, period: str, retrain: bool):
    logger.info("=" * 60)
    logger.info(f"  AI STOCK PREDICTOR — {symbol}  [{interval}]")
    logger.info("=" * 60)

    # ── 1. Fetch Data ─────────────────────────────────────────
    df_raw = fetch_stock_data(symbol=symbol, interval=interval, period=period)

    # ── 2. Feature Engineering ────────────────────────────────
    df = engineer_features(df_raw)

    # ── 3. Preprocess ─────────────────────────────────────────
    X_train, y_train, X_test, y_test, scaler, train_size, target_idx, numeric_df = \
        preprocess_data(df)

    n_features = X_train.shape[2]
    input_shape = (TIME_STEP, n_features)

    # ── 4. Build / Load Model ─────────────────────────────────
    if os.path.exists(MODEL_PATH) and not retrain:
        logger.info(f"Loading existing model from {MODEL_PATH} …")
        model   = load_model(MODEL_PATH)
        history = None
    else:
        logger.info("Building new LSTM model …")
        model = build_model(input_shape)
        history = train_model(model, X_train, y_train)
        save_model(model)

    # ── 5. Predictions ────────────────────────────────────────
    train_pred, test_pred = predict_sets(
        model, X_train, X_test, scaler, target_idx, n_features
    )
    actual_train, actual_test = inverse_actual(
        y_train, y_test, scaler, target_idx, n_features
    )

    # Current price = latest Close in dataset
    current_price = float(df["Close"].iloc[-1])
    scaled_all    = scaler.transform(numeric_df)

    next_day  = predict_next_day(
        model, scaled_all, scaler, target_idx, n_features, current_price
    )
    next_5min = predict_next_5min(
        model, scaled_all, scaler, target_idx, n_features, current_price
    )

    # ── 6. Evaluation ─────────────────────────────────────────
    logger.info("\n📐 EVALUATING ON TEST SET …")
    metrics = evaluate_model(actual_test, test_pred)

    # ── 7. Backtesting ────────────────────────────────────────
    logger.info("\n📈 RUNNING BACKTESTING …")
    bt_result = backtest(actual_test, test_pred)

    # ── 8. Visualize ──────────────────────────────────────────
    logger.info("\n🎨 CREATING VISUALIZATIONS …")
    create_visualizations(
        df, train_pred, test_pred,
        actual_train, actual_test,
        history, bt_result, symbol, train_size,
    )

    # ── 9. Print Final Report ─────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  FINAL PREDICTION REPORT")
    logger.info("=" * 60)
    logger.info(f"  Symbol          : {symbol}")
    logger.info(f"  Current Price   : ₹ {current_price:,.2f}")
    logger.info("")
    logger.info("  ── Next-Day Forecast ──")
    for k, v in next_day.items():
        logger.info(f"    {k:<28}: {v}")
    logger.info("")
    logger.info("  ── 5-Min Simulation ──")
    for k, v in next_5min.items():
        if k == "Simulated 5-Min Prices":
            for step, price in v.items():
                logger.info(f"    {step:<28}: ₹ {price:,.2f}")
        else:
            logger.info(f"    {k:<28}: {v}")
    logger.info("")
    logger.info("  ── Evaluation ──")
    for k, v in metrics.items():
        logger.info(f"    {k:<28}: {v:.4f}")
    logger.info("")
    logger.info("  ── Backtesting ──")
    for k, v in bt_result.items():
        if k not in ("Trade Log", "Portfolio Values"):
            logger.info(f"    {k:<28}: {v}")
    logger.info("=" * 60)

    return {
        "symbol"       : symbol,
        "current_price": current_price,
        "next_day"     : next_day,
        "next_5min"    : next_5min,
        "metrics"      : metrics,
        "backtesting"  : {k:v for k, v in bt_result.items() if k not in ("Trade Log","Portfolio Values")},
    }


# ─────────────────────────────────────────────
#  CLI ARGUMENT PARSER
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Stock Price Prediction System (LSTM)"
    )
    parser.add_argument(
        "--symbol",   type=str, default=DEFAULT_SYMBOL,
        help=f"Ticker symbol (default: {DEFAULT_SYMBOL})"
    )
    parser.add_argument(
        "--interval", type=str, default=DEFAULT_INTERVAL,
        choices=["1d", "5m"],
        help="Data interval: '1d' daily | '5m' intraday (default: 1d)"
    )
    parser.add_argument(
        "--period",   type=str, default=DEFAULT_PERIOD,
        help=f"Look-back period for yfinance (default: {DEFAULT_PERIOD})"
    )
    parser.add_argument(
        "--retrain",  action="store_true",
        help="Force retrain even if model.h5 already exists"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    args   = parse_args()
    result = run_pipeline(
        symbol   = args.symbol,
        interval = args.interval,
        period   = args.period,
        retrain  = args.retrain,
    )

    # Optionally dump result to JSON
    out_json = "prediction_result.json"
    with open(out_json, "w") as f:
        # Convert numpy types to Python native for JSON serialization
        def _convert(obj):
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray):     return obj.tolist()
            return obj
        json.dump(result, f, indent=2, default=_convert)
    logger.info(f"Results saved → {os.path.abspath(out_json)}")

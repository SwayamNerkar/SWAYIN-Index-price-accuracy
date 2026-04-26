"""
predictor.py - Prediction Engine
==================================
Handles:
  1. Train-set predictions  (fit sanity check)
  2. Test-set predictions   (generalisation evaluation)
  3. Next-day price forecast
  4. Next 5-minute price simulation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils import logger, generate_signal
from config import TIME_STEP, NEXT_5MIN_STEPS


# ─────────────────────────────────────────────
#  HELPER: inverse-transform only the Close column
# ─────────────────────────────────────────────
def _inverse_close(
    scaled_values : np.ndarray,
    scaler        : MinMaxScaler,
    target_idx    : int,
    n_features    : int,
) -> np.ndarray:
    """
    Inverse-transform a 1-D array of scaled Close values back to original scale.

    MinMaxScaler expects a full-width (n_features) matrix, so we pad with zeros
    and extract only the target column after inverse transformation.
    """
    dummy          = np.zeros((len(scaled_values), n_features))
    dummy[:, target_idx] = scaled_values.ravel()
    return scaler.inverse_transform(dummy)[:, target_idx]


# ─────────────────────────────────────────────
#  PREDICT TRAIN & TEST SETS
# ─────────────────────────────────────────────
def predict_sets(
    model,
    X_train      : np.ndarray,
    X_test       : np.ndarray,
    scaler       : MinMaxScaler,
    target_idx   : int,
    n_features   : int,
):
    """
    Generate predictions for train and test splits, then inverse-transform them.

    Returns:
        train_pred, test_pred : numpy arrays in original price scale
    """
    logger.info("Generating train-set predictions …")
    train_pred_scaled = model.predict(X_train, verbose=0).ravel()
    train_pred        = _inverse_close(train_pred_scaled, scaler, target_idx, n_features)

    logger.info("Generating test-set predictions …")
    test_pred_scaled  = model.predict(X_test, verbose=0).ravel()
    test_pred         = _inverse_close(test_pred_scaled, scaler, target_idx, n_features)

    logger.info(
        f"Predictions ready | "
        f"train={len(train_pred)} | test={len(test_pred)}"
    )
    return train_pred, test_pred


# ─────────────────────────────────────────────
#  NEXT-DAY PRICE PREDICTION
# ─────────────────────────────────────────────
def predict_next_day(
    model,
    scaled_data  : np.ndarray,
    scaler       : MinMaxScaler,
    target_idx   : int,
    n_features   : int,
    current_price: float,
) -> dict:
    """
    Predict tomorrow's closing price using the last TIME_STEP rows
    of the scaled dataset as the input sequence.

    Args:
        model         : Trained Keras LSTM model.
        scaled_data   : Full scaled feature matrix (shape [rows, features]).
        scaler        : Fitted MinMaxScaler.
        target_idx    : Column index of target ('Close') in scaled_data.
        n_features    : Total number of features.
        current_price : Latest actual close price (for signal generation).

    Returns:
        Dictionary with predicted price, signal, confidence.
    """
    if len(scaled_data) < TIME_STEP:
        raise ValueError(
            f"Not enough data rows ({len(scaled_data)}) "
            f"to form a sequence of length {TIME_STEP}."
        )

    # Take the last TIME_STEP rows as the input window
    last_sequence = scaled_data[-TIME_STEP:]               # shape: (TIME_STEP, features)
    X_input       = last_sequence.reshape(1, TIME_STEP, n_features)

    logger.info("Predicting next-day price …")
    pred_scaled      = model.predict(X_input, verbose=0)[0, 0]
    predicted_price  = _inverse_close(
        np.array([pred_scaled]), scaler, target_idx, n_features
    )[0]

    signal_info = generate_signal(predicted_price, current_price)

    logger.info(f"[Next-Day] Predicted: {predicted_price:.2f} | {signal_info['Signal']}")
    return {
        "Predicted Next-Day Close" : round(predicted_price, 2),
        **signal_info,
    }


# ─────────────────────────────────────────────
#  NEXT 5-MINUTE PRICE SIMULATION
# ─────────────────────────────────────────────
def predict_next_5min(
    model,
    scaled_data  : np.ndarray,
    scaler       : MinMaxScaler,
    target_idx   : int,
    n_features   : int,
    current_price: float,
    steps        : int = NEXT_5MIN_STEPS,
) -> dict:
    """
    Simulate next N × 5-minute price predictions using an
    autoregressive (recursive) forecasting approach:
      - Each prediction is appended to the input window
        (only the target column is updated; other features are held constant)
      - This simulates 'live' 5-minute forecasting when a live data feed
        is unavailable.

    Args:
        model         : Trained Keras LSTM model.
        scaled_data   : Full scaled feature matrix.
        scaler        : Fitted MinMaxScaler.
        target_idx    : Column index of 'Close' in scaled_data.
        n_features    : Total number of features.
        current_price : Latest actual close price.
        steps         : Number of 5-minute steps to predict ahead.

    Returns:
        Dictionary with simulated 5-minute predictions and final signal.
    """
    logger.info(f"Simulating next {steps} × 5-minute predictions …")

    window = scaled_data[-TIME_STEP:].copy()   # shape (TIME_STEP, n_features)
    predictions_scaled = []

    for step in range(steps):
        X_input   = window.reshape(1, TIME_STEP, n_features)
        pred      = model.predict(X_input, verbose=0)[0, 0]
        predictions_scaled.append(pred)

        # Shift window forward: drop oldest row, append new row
        new_row              = window[-1].copy()
        new_row[target_idx]  = pred
        window               = np.vstack([window[1:], new_row])

    # Inverse-transform predicted values
    pred_prices = _inverse_close(
        np.array(predictions_scaled), scaler, target_idx, n_features
    )

    # Use the last step's predicted price for signal generation
    final_pred  = float(pred_prices[-1])
    signal_info = generate_signal(final_pred, current_price)

    step_labels = [f"t+{(i+1)*5}min" for i in range(steps)]
    simulation  = dict(zip(step_labels, [round(p, 2) for p in pred_prices]))

    logger.info(f"[5-Min Sim] Steps: {simulation} | {signal_info['Signal']}")

    return {
        "Simulated 5-Min Prices" : simulation,
        "Final Predicted Price"  : round(final_pred, 2),
        **signal_info,
    }


# ─────────────────────────────────────────────
#  INVERSE TRANSFORM ACTUAL VALUES
# ─────────────────────────────────────────────
def inverse_actual(
    y_train      : np.ndarray,
    y_test       : np.ndarray,
    scaler       : MinMaxScaler,
    target_idx   : int,
    n_features   : int,
):
    """
    Inverse-transform the actual y labels for comparison with predictions.
    Returns:
        actual_train, actual_test : numpy arrays in original price scale
    """
    actual_train = _inverse_close(y_train, scaler, target_idx, n_features)
    actual_test  = _inverse_close(y_test,  scaler, target_idx, n_features)
    return actual_train, actual_test

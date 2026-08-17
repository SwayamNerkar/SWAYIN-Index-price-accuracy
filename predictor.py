"""
predictor.py - Backward Compatibility Adapter for Backend Prediction Engine
"""

import numpy as np
import pandas as pd
from backend.app.services.buy_sell_engine import generate_trading_signal
from config import TIME_STEP, NEXT_5MIN_STEPS

def _inverse_close(scaled_values, scaler, target_idx, n_features):
    dummy = np.zeros((len(scaled_values), n_features))
    dummy[:, target_idx] = scaled_values.ravel()
    return scaler.inverse_transform(dummy)[:, target_idx]

def predict_sets(model, X_train, X_test, scaler, target_idx, n_features):
    train_pred_scaled = model.predict(X_train, verbose=0).ravel()
    train_pred = _inverse_close(train_pred_scaled, scaler, target_idx, n_features)
    test_pred_scaled = model.predict(X_test, verbose=0).ravel()
    test_pred = _inverse_close(test_pred_scaled, scaler, target_idx, n_features)
    return train_pred, test_pred

def predict_next_day(model, scaled_data, scaler, target_idx, n_features, current_price):
    time_step = model.input_shape[1] if hasattr(model, 'input_shape') and model.input_shape[1] is not None else TIME_STEP
    time_step = min(time_step, len(scaled_data))
    
    last_seq = scaled_data[-time_step:].reshape(1, time_step, n_features)
    pred_scaled = model.predict(last_seq, verbose=0)[0, 0]
    pred_price = float(_inverse_close(np.array([pred_scaled]), scaler, target_idx, n_features)[0])
    
    chg_pct = ((pred_price - current_price) / current_price) * 100.0
    sig = "BUY 📈" if chg_pct > 0 else "SELL 📉"
    conf = min(abs(chg_pct) * 10.0 + 75.0, 99.0)

    return {
        "Predicted Next-Day Close": round(pred_price, 2),
        "Signal": sig,
        "Current Price": round(current_price, 2),
        "Predicted Price": round(pred_price, 2),
        "Change (%)": round(chg_pct, 2),
        "Confidence (%)": round(conf, 2)
    }

def predict_next_5min(model, scaled_data, scaler, target_idx, n_features, current_price, steps=NEXT_5MIN_STEPS):
    time_step = model.input_shape[1] if hasattr(model, 'input_shape') and model.input_shape[1] is not None else TIME_STEP
    time_step = min(time_step, len(scaled_data))

    window = scaled_data[-time_step:].copy()
    preds_scaled = []
    for _ in range(steps):
        X_input = window.reshape(1, time_step, n_features)
        pred = model.predict(X_input, verbose=0)[0, 0]
        preds_scaled.append(pred)
        new_row = window[-1].copy()
        new_row[target_idx] = pred
        window = np.vstack([window[1:], new_row])

    pred_prices = _inverse_close(np.array(preds_scaled), scaler, target_idx, n_features)
    final_pred = float(pred_prices[-1])
    chg_pct = ((final_pred - current_price) / current_price) * 100.0
    sig = "BUY 📈" if chg_pct > 0 else "SELL 📉"
    conf = min(abs(chg_pct) * 10.0 + 75.0, 99.0)

    sim = {f"t+{(i+1)*5}min": round(float(p), 2) for i, p in enumerate(pred_prices)}

    return {
        "Simulated 5-Min Prices": sim,
        "Final Predicted Price": round(final_pred, 2),
        "Signal": sig,
        "Current Price": round(current_price, 2),
        "Predicted Price": round(final_pred, 2),
        "Change (%)": round(chg_pct, 2),
        "Confidence (%)": round(conf, 2)
    }

def inverse_actual(y_train, y_test, scaler, target_idx, n_features):
    act_train = _inverse_close(y_train, scaler, target_idx, n_features)
    act_test = _inverse_close(y_test, scaler, target_idx, n_features)
    return act_train, act_test

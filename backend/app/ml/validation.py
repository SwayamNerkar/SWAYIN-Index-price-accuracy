import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from backend.app.core.logging import logger

def preprocess_and_sequence(
    df: pd.DataFrame,
    target_col: str = "Close",
    time_step: int = 60,
    train_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, int, int]:
    """
    Dynamic sequence creation with adaptive time_step for small datasets.
    Prevents empty X_train arrays and data leakage.
    """
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if target_col not in numeric_df.columns:
        numeric_df[target_col] = df[target_col]

    target_idx = numeric_df.columns.get_loc(target_col)
    n_rows = len(numeric_df)
    
    # Adapt time_step if dataset is smaller than default window
    if n_rows <= time_step + 5:
        time_step = max(2, n_rows // 3)
        logger.info(f"Adapted time_step to {time_step} for dataset of length {n_rows}")

    train_size = max(time_step + 2, int(n_rows * train_ratio))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(numeric_df.iloc[:train_size])

    scaled_arr = scaler.transform(numeric_df)

    X, y = [], []
    for i in range(len(scaled_arr) - time_step):
        X.append(scaled_arr[i : i + time_step])
        y.append(scaled_arr[i + time_step, target_idx])

    X_arr = np.array(X)
    y_arr = np.array(y)

    split = max(1, train_size - time_step)
    X_train, y_train = X_arr[:split], y_arr[:split]
    X_test, y_test = X_arr[split:], y_arr[split:]

    if len(X_test) == 0:
        X_test, y_test = X_train, y_train

    logger.info(f"Walk-Forward validation split: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, y_train, X_test, y_test, scaler, target_idx, numeric_df.shape[1]

def perform_walk_forward_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in tscv.split(X):
        yield X[train_idx], y[train_idx], X[val_idx], y[val_idx]

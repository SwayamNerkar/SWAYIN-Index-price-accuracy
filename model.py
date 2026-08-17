"""
model.py - Backward Compatibility Adapter for Backend ML Models
"""

import os
import joblib
import numpy as np
import pandas as pd
from backend.app.ml.validation import preprocess_and_sequence
from backend.app.core.config import settings
from backend.app.core.logging import logger
from config import (
    TIME_STEP,
    TRAIN_RATIO,
    LSTM_UNITS,
    DROPOUT_RATE,
    DENSE_UNITS,
    OPTIMIZER,
    LOSS,
    EPOCHS,
    BATCH_SIZE,
    MODEL_PATH,
)

def preprocess_data(df: pd.DataFrame, feature_col: str = "Close"):
    X_train, y_train, X_test, y_test, scaler, target_idx, n_features = preprocess_and_sequence(
        df, target_col=feature_col, time_step=TIME_STEP
    )
    train_size = int(len(df) * 0.8)
    numeric_df = df.select_dtypes(include=[np.number])
    return X_train, y_train, X_test, y_test, scaler, train_size, target_idx, numeric_df

def build_model(input_shape: tuple):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        Dropout(0.15),
        LSTM(128, return_sequences=True),
        Dropout(0.15),
        LSTM(64, return_sequences=False),
        Dropout(0.15),
        Dense(25, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="huber", metrics=["mae"])
    return model

def train_model(model, X_train: np.ndarray, y_train: np.ndarray):
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6)
    ]
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=0.10,
        callbacks=callbacks,
        verbose=0
    )
    return history

def save_model(model, path: str = MODEL_PATH):
    model.save(path)

def load_model(path: str = MODEL_PATH):
    from tensorflow.keras.models import load_model as _load
    return _load(path)

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils import logger
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
    """
    Scale data with MinMaxScaler and split into train / test sets.

    Args:
        df          : Feature-engineered DataFrame.
        feature_col : Target column to predict (default 'Close').

    Returns:
        X_train, y_train, X_test, y_test : numpy arrays
        scaler                            : fitted MinMaxScaler (needed for inverse_transform)
        train_size                        : int — number of training rows
    """
    # ── Select only numeric columns for scaling ───────────────
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # ── Fit scaler on training portion only (no data leakage) ─
    train_size = int(len(numeric_df) * TRAIN_RATIO)
    scaler     = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(numeric_df.iloc[:train_size])

    scaled = scaler.transform(numeric_df)

    # Index of the target column in scaled array
    target_idx = numeric_df.columns.get_loc(feature_col)

    # ── Create time-series sequences ──────────────────────────
    X, y = _create_sequences(scaled, target_idx, TIME_STEP)

    # ── Train / test split ────────────────────────────────────
    split    = train_size - TIME_STEP
    X_train  = X[:split]
    y_train  = y[:split]
    X_test   = X[split:]
    y_test   = y[split:]

    logger.info(
        f"Preprocessing complete | "
        f"TIME_STEP={TIME_STEP} | "
        f"X_train={X_train.shape} | X_test={X_test.shape}"
    )
    return X_train, y_train, X_test, y_test, scaler, train_size, target_idx, numeric_df


def _create_sequences(scaled: np.ndarray, target_idx: int, time_step: int):
    """
    Convert scaled 2-D array into supervised learning sequences.

    X[i] = scaled[i : i+time_step]          (all features, shape: time_step × features)
    y[i] = scaled[i+time_step, target_idx]   (target value at next step)
    """
    X, y = [], []
    for i in range(len(scaled) - time_step):
        X.append(scaled[i : i + time_step])
        y.append(scaled[i + time_step, target_idx])
    return np.array(X), np.array(y)

def build_model(input_shape: tuple):
 
    # Import inside function to avoid slow startup when not training
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    model = Sequential(name="DeepLSTM_StockPredictor")

    # ── Layer 1: LSTM 100 (returns sequences for stacking) ────
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=LSTM_UNITS[0], return_sequences=True, name="lstm_1"))
    model.add(Dropout(DROPOUT_RATE, name="dropout_1"))

    # ── Layer 2: LSTM 100 (returns sequences for stacking) ────
    model.add(LSTM(units=LSTM_UNITS[1], return_sequences=True, name="lstm_2"))
    model.add(Dropout(DROPOUT_RATE, name="dropout_2"))

    # ── Layer 3: LSTM 50 (final recurrent layer) ───────────────
    model.add(LSTM(units=LSTM_UNITS[2], return_sequences=False, name="lstm_3"))
    model.add(Dropout(DROPOUT_RATE, name="dropout_3"))

    # ── Dense head ────────────────────────────────────────────
    model.add(Dense(25, activation="relu", name="dense_hidden"))
    model.add(Dense(DENSE_UNITS, name="dense_output"))

    from tensorflow.keras.optimizers import Adam
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=LOSS, metrics=["mae"])

    logger.info(f"Model built: input_shape={input_shape}")
    model.summary(print_fn=logger.info)

    return model

def train_model(model, X_train: np.ndarray, y_train: np.ndarray):
    """
    Train the LSTM model with early stopping and LR scheduling.

    Args:
        model   : Compiled Keras model (from build_model).
        X_train : Training sequences.
        y_train : Training labels.

    Returns:
        Keras History object.
    """
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,           # Reduced patience for 25 epochs
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,           # Reduced patience for faster adaptation
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    logger.info(
        f"Training | epochs={EPOCHS} | batch_size={BATCH_SIZE} | "
        f"validation_split=0.10"
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.10,
        callbacks=callbacks,
        verbose=1,
    )

    logger.info(f"Training complete. Best val_loss: {min(history.history['val_loss']):.6f}")
    return history

def save_model(model, path: str = MODEL_PATH) -> None:
    """Save trained model to .h5 file."""
    model.save(path)
    logger.info(f"Model saved → {os.path.abspath(path)}")


def load_model(path: str = MODEL_PATH):
    """Load a previously saved .h5 model."""
    from tensorflow.keras.models import load_model as _load
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = _load(path)
    logger.info(f"Model loaded ← {os.path.abspath(path)}")
    return model

import os
import numpy as np
from typing import Dict, Any
from backend.app.core.logging import logger

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        LSTM, Bidirectional, GRU, Conv1D, MaxPooling1D,
        Dense, Dropout, Input, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TF = True
except ImportError:
    HAS_TF = False

class DeepLearningSuite:
    """
    Suite of Deep Learning models for time-series forecasting:
    - Stacked LSTM
    - Bidirectional LSTM (BiLSTM)
    - GRU
    - Hybrid CNN-LSTM
    - Time-Series Transformer
    """
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape
        self.models = {}
        if HAS_TF:
            self._build_models()

    def _build_models(self):
        # 1. Stacked LSTM
        lstm = Sequential([
            Input(shape=self.input_shape),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1)
        ], name="LSTM")
        lstm.compile(optimizer="adam", loss="huber", metrics=["mae"])
        self.models["LSTM"] = lstm

        # 2. Bidirectional LSTM (BiLSTM)
        bilstm = Sequential([
            Input(shape=self.input_shape),
            Bidirectional(LSTM(32, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(16, return_sequences=False)),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1)
        ], name="BiLSTM")
        bilstm.compile(optimizer="adam", loss="huber", metrics=["mae"])
        self.models["BiLSTM"] = bilstm

        # 3. GRU Model
        gru = Sequential([
            Input(shape=self.input_shape),
            GRU(64, return_sequences=True),
            Dropout(0.2),
            GRU(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1)
        ], name="GRU")
        gru.compile(optimizer="adam", loss="huber", metrics=["mae"])
        self.models["GRU"] = gru

        # 4. CNN-LSTM Model
        cnn_lstm = Sequential([
            Input(shape=self.input_shape),
            Conv1D(filters=32, kernel_size=2, activation="relu", padding="same"),
            MaxPooling1D(pool_size=1),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1)
        ], name="CNN_LSTM")
        cnn_lstm.compile(optimizer="adam", loss="huber", metrics=["mae"])
        self.models["CNN-LSTM"] = cnn_lstm

        # 5. Time Series Transformer Model
        inputs = Input(shape=self.input_shape)
        x = LayerNormalization(epsilon=1e-6)(inputs)
        attn_output = MultiHeadAttention(key_dim=32, num_heads=2, dropout=0.1)(x, x)
        x = x + attn_output
        x = LayerNormalization(epsilon=1e-6)(x)
        x_ffn = Dense(64, activation="relu")(x)
        x_ffn = Dense(self.input_shape[1])(x_ffn)
        x = x + x_ffn
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1)(x)
        transformer = Model(inputs=inputs, outputs=outputs, name="Transformer")
        transformer.compile(optimizer="adam", loss="huber", metrics=["mae"])
        self.models["Transformer"] = transformer

    def train_model(self, name: str, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 15, batch_size: int = 32):
        if not HAS_TF or name not in self.models or len(X_train) == 0:
            return None
            
        val_split = 0.1 if len(X_train) >= 20 else 0.0
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)
        ] if val_split > 0 else []
        
        logger.info(f"Training Deep Learning model: {name}")
        history = self.models[name].fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            callbacks=callbacks,
            verbose=0
        )
        return history

    def train_all(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 15, batch_size: int = 32):
        histories = {}
        for name in self.models:
            hist = self.train_model(name, X_train, y_train, epochs=epochs, batch_size=batch_size)
            histories[name] = hist
        return histories

    def predict_all(self, X_input: np.ndarray) -> Dict[str, np.ndarray]:
        preds = {}
        for name, model in self.models.items():
            try:
                preds[name] = model.predict(X_input, verbose=0).ravel()
            except Exception as e:
                logger.error(f"Error predicting with {name}: {e}")
                preds[name] = np.zeros(len(X_input))
        return preds

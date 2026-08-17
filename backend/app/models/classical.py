import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from backend.app.core.logging import logger

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

class ClassicalModelSuite:
    """
    Suite of classical ML and Gradient Boosted Decision Tree models:
    - Random Forest
    - XGBoost
    - LightGBM
    - CatBoost
    - Ridge (Linear)
    - Gradient Boosting
    """
    def __init__(self):
        self.models = {}
        self._init_models()

    def _init_models(self):
        self.models["Random Forest"] = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.models["Linear Regression"] = Ridge(alpha=1.0)
        self.models["Gradient Boosting"] = GradientBoostingRegressor(n_estimators=100, random_state=42)

        if HAS_XGBOOST:
            self.models["XGBoost"] = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)
        else:
            self.models["XGBoost"] = RandomForestRegressor(n_estimators=80, random_state=42)

        if HAS_LIGHTGBM:
            self.models["LightGBM"] = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbosity=-1)
        else:
            self.models["LightGBM"] = GradientBoostingRegressor(n_estimators=80, random_state=42)

        if HAS_CATBOOST:
            self.models["CatBoost"] = cb.CatBoostRegressor(iterations=100, learning_rate=0.05, verbose=0, random_seed=42)
        else:
            self.models["CatBoost"] = RandomForestRegressor(n_estimators=80, random_state=42)

    def train_all(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        results = {}
        # Flatten sequence input if 3D
        if len(X_train.shape) == 3:
            nsamples, nx, ny = X_train.shape
            X_train_flat = X_train.reshape((nsamples, nx * ny))
        else:
            X_train_flat = X_train

        for name, model in self.models.items():
            try:
                model.fit(X_train_flat, y_train)
                results[name] = "Success"
                logger.info(f"Trained classical model: {name}")
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                results[name] = f"Failed: {e}"
        return results

    def predict_all(self, X_input: np.ndarray) -> Dict[str, np.ndarray]:
        if len(X_input.shape) == 3:
            nsamples, nx, ny = X_input.shape
            X_flat = X_input.reshape((nsamples, nx * ny))
        else:
            X_flat = X_input

        preds = {}
        for name, model in self.models.items():
            try:
                preds[name] = model.predict(X_flat)
            except Exception as e:
                logger.error(f"Error in prediction for {name}: {e}")
                preds[name] = np.zeros(len(X_flat))
        return preds

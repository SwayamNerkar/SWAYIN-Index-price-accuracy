import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.ml.validation import preprocess_and_sequence
from backend.app.models.classical import ClassicalModelSuite
from backend.app.models.deep_learning import DeepLearningSuite, HAS_TF
from backend.app.services.ensemble import calculate_metrics, DynamicEnsembleEngine

class ModelTrainerPipeline:
    """
    Master Training Pipeline:
    Coordinates Classical models, GBDT models, Deep Learning models,
    Walk-Forward validation metrics, model versioning, and disk persistence.
    """
    def __init__(self, time_step: int = 60):
        self.time_step = time_step
        self.ensemble_engine = DynamicEnsembleEngine()

    def train_pipeline(self, df: pd.DataFrame, epochs: int = 3) -> Dict[str, Any]:
        logger.info(f"Starting Master Model Training Pipeline (epochs={epochs})...")
        X_train, y_train, X_test, y_test, scaler, target_idx, n_features = preprocess_and_sequence(
            df, time_step=self.time_step
        )

        input_shape = (self.time_step, n_features)
        val_metrics = {}
        all_test_preds = {}

        # 1. Classical & GBDT Models
        classical_suite = ClassicalModelSuite()
        classical_suite.train_all(X_train, y_train)
        classical_preds = classical_suite.predict_all(X_test)

        for name, pred in classical_preds.items():
            metrics = calculate_metrics(y_test, pred)
            val_metrics[name] = metrics
            all_test_preds[name] = pred

        # 2. Deep Learning Models (fast fitting for responsive UI)
        dl_suite = DeepLearningSuite(input_shape)
        if HAS_TF:
            dl_suite.train_all(X_train, y_train, epochs=epochs, batch_size=64)
            dl_preds = dl_suite.predict_all(X_test)
            for name, pred in dl_preds.items():
                metrics = calculate_metrics(y_test, pred)
                val_metrics[name] = metrics
                all_test_preds[name] = pred

        # 3. Dynamic Ensemble Weights
        weights = self.ensemble_engine.compute_weights(val_metrics)

        # 4. Save Models & Scaler
        version_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        bundle_path = os.path.join(settings.SAVED_MODELS_DIR, f"model_bundle_{version_str}.joblib")
        latest_path = os.path.join(settings.SAVED_MODELS_DIR, "latest_bundle.joblib")

        save_dict = {
            "version": version_str,
            "scaler": scaler,
            "target_idx": target_idx,
            "n_features": n_features,
            "time_step": self.time_step,
            "weights": weights,
            "val_metrics": val_metrics,
            "classical_suite": classical_suite,
            "dl_suite": dl_suite if HAS_TF else None
        }

        joblib.dump(save_dict, bundle_path)
        joblib.dump(save_dict, latest_path)
        logger.info(f"Model bundle saved -> {bundle_path}")

        return {
            "version": version_str,
            "bundle_path": bundle_path,
            "metrics": val_metrics,
            "weights": weights
        }

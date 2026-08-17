import numpy as np
from typing import Dict, Any, List
from backend.app.core.logging import logger

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    
    actual_dir = np.diff(y_true) > 0
    pred_dir = np.diff(y_pred) > 0
    dir_acc = float(np.mean(actual_dir == pred_dir) * 100) if len(actual_dir) > 0 else 50.0

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Directional_Accuracy": dir_acc
    }

class DynamicEnsembleEngine:
    """
    Dynamic performance-weighted ensemble engine combining predictions from all
    Classical, GBDT, and Deep Learning models based on validation performance.
    """
    def __init__(self):
        self.model_weights = {}

    def compute_weights(self, validation_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Compute normalized model weights based on inverse RMSE and Directional Accuracy.
        """
        raw_scores = {}
        for name, metrics in validation_results.items():
            rmse = metrics.get("RMSE", 1.0)
            dir_acc = metrics.get("Directional_Accuracy", 50.0) / 100.0
            
            # Score formula: (1 / (RMSE + 1e-5)) * (Directional Accuracy ^ 2)
            score = (1.0 / (rmse + 1e-5)) * (dir_acc ** 2)
            raw_scores[name] = max(score, 1e-6)

        total_score = sum(raw_scores.values())
        self.model_weights = {name: score / total_score for name, score in raw_scores.items()}
        return self.model_weights

    def predict_ensemble(
        self,
        individual_predictions: Dict[str, float],
        weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Generate weighted ensemble prediction and consensus metrics.
        """
        if not weights:
            weights = self.model_weights
            
        if not weights:
            # Equal weighting fallback
            n = len(individual_predictions)
            weights = {k: 1.0 / n for k in individual_predictions}

        weighted_price = 0.0
        total_w = 0.0
        
        preds_list = []
        for name, pred in individual_predictions.items():
            w = weights.get(name, 0.1)
            weighted_price += pred * w
            total_w += w
            preds_list.append(pred)

        final_predicted_price = weighted_price / total_w if total_w > 0 else np.mean(preds_list)
        
        # Calculate model variance and confidence spread
        pred_std = float(np.std(preds_list))
        ensemble_confidence = max(50.0, min(99.0, 100.0 - (pred_std / final_predicted_price * 1000.0)))

        return {
            "ensemble_predicted_price": round(float(final_predicted_price), 2),
            "model_weights": weights,
            "predictions_breakdown": {k: round(float(v), 2) for k, v in individual_predictions.items()},
            "prediction_std": round(pred_std, 2),
            "ensemble_confidence": round(ensemble_confidence, 2)
        }

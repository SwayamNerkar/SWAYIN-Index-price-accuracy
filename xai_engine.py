"""
xai_engine.py - Backward Compatibility Adapter for Backend XAI Sensitivity Engine
"""

import numpy as np
from backend.app.core.logging import logger

def get_feature_importance(model, input_sequence, feature_names):
    """
    Sensitivity analysis feature importance calculation.
    """
    try:
        base_pred = model.predict(input_sequence, verbose=0)[0][0]
        importances = {}
        n_features = input_sequence.shape[2]
        epsilon = 0.05
        
        for i in range(n_features):
            pert = input_sequence.copy()
            pert[0, :, i] += epsilon
            p_pred = model.predict(pert, verbose=0)[0][0]
            importances[feature_names[i]] = float(abs(p_pred - base_pred))

        total = sum(importances.values()) if sum(importances.values()) > 0 else 1.0
        norm_imp = {k: (v / total) * 100.0 for k, v in importances.items()}
        return dict(sorted(norm_imp.items(), key=lambda item: item[1], reverse=True))
    except Exception as e:
        logger.error(f"[XAI] Feature importance fallback: {e}")
        return {name: 100.0 / len(feature_names) for name in feature_names}

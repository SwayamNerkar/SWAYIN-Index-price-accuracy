import numpy as np
import pandas as pd
from utils import logger

def get_feature_importance(model, input_sequence, feature_names):
    """
    Calculate feature importance using a sensitivity analysis approach.
    Perturbs each feature in the input sequence and measures the impact on output.
    
    Args:
        model           : The loaded Keras LSTM model.
        input_sequence  : A single input sequence (shape: 1, TIME_STEP, n_features).
        feature_names   : List of strings corresponding to the features.
        
    Returns:
        dict: { 'feature_name': importance_score }
    """
    try:
        # Base prediction
        base_pred = model.predict(input_sequence, verbose=0)[0][0]
        
        importances = {}
        n_features = input_sequence.shape[2]
        
        # We'll perturb features by a small epsilon value
        epsilon = 0.05 
        
        for i in range(n_features):
            # Create a copy and perturb the i-th feature
            perturbed_seq = input_sequence.copy()
            
            # Add epsilon to the entire column of the feature in the sequence
            perturbed_seq[0, :, i] += epsilon
            
            # Predict with perturbed input
            perturbed_pred = model.predict(perturbed_seq, verbose=0)[0][0]
            
            # Importance = Absolute change in prediction
            # We use absolute because both increase and decrease show influence
            change = abs(perturbed_pred - base_pred)
            importances[feature_names[i]] = float(change)
            
        # Normalize to sum to 100%
        total = sum(importances.values()) if sum(importances.values()) > 0 else 1
        normalized_importances = {k: (v / total) * 100 for k, v in importances.items()}
        
        # Sort by importance
        sorted_importances = dict(sorted(normalized_importances.items(), key=lambda item: item[1], reverse=True))
        
        logger.info(f"[XAI] Feature Importance calculated for {len(feature_names)} features.")
        return sorted_importances

    except Exception as e:
        logger.error(f"[XAI] ❌ Feature importance calculation failed: {e}")
        return {name: 100.0/len(feature_names) for name in feature_names}

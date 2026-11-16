"""
Production Monitoring and Drift Detection Module.
Monitors Population Stability Index (PSI) and feature distributions against training baseline.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from src.metrics import calculate_psi


class CreditRiskMonitor:
    """
    Monitors data drift and model score drift against a baseline distribution.
    """

    def __init__(self, baseline_df: pd.DataFrame, baseline_probs: np.ndarray):
        self.baseline_df = baseline_df.copy()
        self.baseline_probs = np.asarray(baseline_probs)

    def assess_score_drift(self, current_probs: np.ndarray, num_buckets: int = 10) -> Dict[str, Any]:
        """Assess drift of model output probabilities using PSI."""
        psi, bucket_df = calculate_psi(self.baseline_probs, current_probs, num_buckets=num_buckets)

        if psi < 0.10:
            status = "STABLE"
            action = "No action required."
        elif psi < 0.25:
            status = "MODERATE_SHIFT"
            action = "Monitor distribution closely; investigate potential macro changes."
        else:
            status = "SIGNIFICANT_DRIFT"
            action = "Significant drift detected. Retraining workflow recommended."

        return {
            "metric": "Population Stability Index (PSI)",
            "psi_score": round(psi, 4),
            "drift_status": status,
            "recommended_action": action,
            "bucket_breakdown": bucket_df.to_dict(orient="records"),
        }

    def assess_feature_drift(self, current_df: pd.DataFrame, features: Optional[List[str]] = None) -> Dict[str, Any]:
        """Assess drift across individual input features."""
        features_to_check = features or [c for c in self.baseline_df.columns if c in current_df.columns]
        feature_results = {}

        for feat in features_to_check:
            base_vals = self.baseline_df[feat].dropna().values
            curr_vals = current_df[feat].dropna().values
            if len(base_vals) > 0 and len(curr_vals) > 0:
                psi, _ = calculate_psi(base_vals, curr_vals, num_buckets=5)
                feature_results[feat] = {
                    "psi": round(psi, 4),
                    "status": "STABLE" if psi < 0.10 else ("MODERATE_DRIFT" if psi < 0.25 else "HIGH_DRIFT"),
                }

        return feature_results

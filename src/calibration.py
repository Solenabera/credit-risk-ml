"""
Probability Calibration Module for Credit Risk Scoring.
Calibrates binary classifier output to empirical default probabilities using
Isotonic Regression or Platt Scaling (Sigmoid) on validation data.
"""

from typing import Optional, Literal
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class CalibratedBinaryClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper that calibrates the predicted probabilities of a binary classification estimator
    using empirical validation data.
    
    In credit risk, uncalibrated models (e.g. tree models trained with scale_pos_weight)
    output distorted probabilities. This calibrator aligns predict_proba with the true
    empirical Probability of Default (PD).
    """

    def __init__(
        self,
        base_estimator,
        method: Literal["isotonic", "sigmoid"] = "isotonic",
    ):
        self.base_estimator = base_estimator
        self.method = method
        self.calibrator_ = None
        self.classes_ = np.array([0, 1])
        self.is_fitted_ = False

    def fit(self, X=None, y=None):
        """No-op fit to satisfy scikit-learn estimator interface."""
        self.is_fitted_ = True
        return self

    def __sklearn_is_fitted__(self) -> bool:
        """Indicate whether the estimator is fitted."""
        return self.is_fitted_ or self.calibrator_ is not None

    def fit_calibrator(self, X_val: np.ndarray, y_val: np.ndarray):
        """Fit the calibrator mapping on the holdout validation set."""
        raw_probs = self.base_estimator.predict_proba(X_val)[:, 1]
        y_val_arr = np.asarray(y_val).astype(int)

        if self.method == "isotonic":
            self.calibrator_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator_.fit(raw_probs, y_val_arr)
        elif self.method == "sigmoid":
            # Platt scaling: univariate logistic regression on raw probabilities
            lr = LogisticRegression(C=1.0, solver="lbfgs")
            lr.fit(raw_probs.reshape(-1, 1), y_val_arr)
            self.calibrator_ = lr
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.is_fitted_ = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return calibrated class probabilities."""
        raw_probs = self.base_estimator.predict_proba(X)[:, 1]

        if self.calibrator_ is None:
            calibrated = raw_probs
        elif self.method == "isotonic":
            calibrated = self.calibrator_.predict(raw_probs)
        elif self.method == "sigmoid":
            calibrated = self.calibrator_.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
        else:
            calibrated = raw_probs

        calibrated = np.clip(calibrated, 0.0, 1.0)
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class given a decision threshold."""
        prob_default = self.predict_proba(X)[:, 1]
        return (prob_default >= threshold).astype(int)

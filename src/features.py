import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

class CreditRiskFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to cap outliers and engineer new features.
    """
    def __init__(self):
        # We can pass hyperparameters here if needed (e.g., capping thresholds)
        pass
        
    def fit(self, X, y=None):
        # fit() is used to learn statistics from the training data (like mean or max).
        # Since our logic relies on hardcoded business rules (not learned stats), we just return self.
        return self
        
    def transform(self, X):
        # Always operate on a copy to avoid SettingWithCopy warnings and unintended mutations
        X_out = X.copy()
        
        # 1. Cap Revolving Utilization (Business rule: utilization rarely exceeds 200% or 2.0)
        if 'RevolvingUtilizationOfUnsecuredLines' in X_out.columns:
            X_out['RevolvingUtilizationOfUnsecuredLines'] = np.where(
                X_out['RevolvingUtilizationOfUnsecuredLines'] > 2.0, 
                2.0, 
                X_out['RevolvingUtilizationOfUnsecuredLines']
            )
            
        # 2. Cap Age (Data entry errors: replace age 0 with the dataset median or a logical minimum)
        if 'age' in X_out.columns:
            X_out['age'] = np.where(X_out['age'] < 18, 18, X_out['age'])
            
        # 3. Engineer Composite Feature: Total Past Due Events
        past_due_cols = [
            'NumberOfTime30-59DaysPastDueNotWorse', 
            'NumberOfTime60-89DaysPastDueNotWorse', 
            'NumberOfTimes90DaysLate'
        ]
        if all(col in X_out.columns for col in past_due_cols):
            X_out['TotalPastDue'] = X_out[past_due_cols].sum(axis=1)
            
        return X_out

print("Custom transformer 'CreditRiskFeatureEngineer' initialized successfully.")
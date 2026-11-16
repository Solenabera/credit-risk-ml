"""
Feature Engineering and Preprocessing Transformers for Credit Risk Scoring.
Implements leak-free, scikit-learn compatible custom transformers with domain-specific rules.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class CreditRiskFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Production-grade Scikit-Learn transformer for Credit Risk feature engineering.
    
    Transforms raw applicant attributes into risk-indicative features:
    1. Replaces delinquency anomaly codes (96, 98) with logical zeros and adds anomaly flag.
    2. Caps extreme RevolvingUtilizationOfUnsecuredLines to prevent leverage outliers.
    3. Enforces biological age minimums (age >= 18).
    4. Creates missing-data indicators for MonthlyIncome and NumberOfDependents.
    5. Computes composite risk signals: TotalPastDue, HasPastDue, EstimatedMonthlyDebt, and PerCapitaIncome.
    """

    def __init__(
        self,
        revolving_utilization_cap: float = 2.0,
        age_min: int = 18,
        debt_ratio_cap: float = 10.0,
        delinquency_cols: Optional[List[str]] = None,
        delinquency_anomaly_codes: Optional[List[int]] = None,
    ):
        self.revolving_utilization_cap = revolving_utilization_cap
        self.age_min = age_min
        self.debt_ratio_cap = debt_ratio_cap
        self.delinquency_cols = delinquency_cols or [
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
        ]
        self.delinquency_anomaly_codes = delinquency_anomaly_codes or [96, 98]
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        """Fit transformer and determine output feature names."""
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        transformed = self._transform_df(df.copy())
        self.feature_names_out_ = list(transformed.columns)
        return self

    def _transform_df(self, X_out: pd.DataFrame) -> pd.DataFrame:
        """Apply domain transformations to DataFrame."""
        # 1. Missingness indicators before imputation
        if "MonthlyIncome" in X_out.columns:
            X_out["MonthlyIncome_Missing"] = X_out["MonthlyIncome"].isnull().astype(float)
        if "NumberOfDependents" in X_out.columns:
            X_out["NumberOfDependents_Missing"] = X_out["NumberOfDependents"].isnull().astype(float)

        # 2. Sanitize delinquency anomaly codes (96, 98 are missing/error codes in credit bureau exports)
        anomaly_detected = pd.Series(False, index=X_out.index)
        for col in self.delinquency_cols:
            if col in X_out.columns:
                is_anomaly = X_out[col].isin(self.delinquency_anomaly_codes)
                anomaly_detected = anomaly_detected | is_anomaly
                # Clean the column: replace anomaly codes with 0 to prevent skewed arithmetic
                X_out[col] = np.where(is_anomaly, 0, X_out[col])
        X_out["DelinquencyDataAnomaly"] = anomaly_detected.astype(float)

        # 3. Cap Revolving Utilization
        if "RevolvingUtilizationOfUnsecuredLines" in X_out.columns:
            X_out["RevolvingUtilizationOfUnsecuredLines"] = np.where(
                X_out["RevolvingUtilizationOfUnsecuredLines"] > self.revolving_utilization_cap,
                self.revolving_utilization_cap,
                X_out["RevolvingUtilizationOfUnsecuredLines"],
            )

        # 4. Enforce Age Floor
        if "age" in X_out.columns:
            X_out["age"] = np.where(X_out["age"] < self.age_min, self.age_min, X_out["age"])

        # 5. Composite Delinquency Metrics
        if all(col in X_out.columns for col in self.delinquency_cols):
            X_out["TotalPastDue"] = X_out[self.delinquency_cols].sum(axis=1)
            X_out["HasPastDue"] = (X_out["TotalPastDue"] > 0).astype(float)

        # 6. Interaction: Per Capita Income & Estimated Debt
        if "MonthlyIncome" in X_out.columns and "NumberOfDependents" in X_out.columns:
            income = X_out["MonthlyIncome"].fillna(0)
            dependents = X_out["NumberOfDependents"].fillna(0)
            X_out["IncomePerCapita"] = income / (dependents + 1.0)

        if "DebtRatio" in X_out.columns and "MonthlyIncome" in X_out.columns:
            # If MonthlyIncome is present, DebtRatio * MonthlyIncome is dollar debt
            income = X_out["MonthlyIncome"].fillna(0)
            debt_ratio = X_out["DebtRatio"].fillna(0)
            X_out["EstimatedDebtPayment"] = debt_ratio * income

        return X_out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform input data."""
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return self._transform_df(df)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return names of output features for pipeline inspection."""
        return np.array(self.feature_names_out_)


def build_preprocessing_pipeline(
    revolving_utilization_cap: float = 2.0,
    age_min: int = 18,
    debt_ratio_cap: float = 10.0,
) -> Pipeline:
    """
    Construct the Scikit-Learn preprocessing pipeline including custom feature engineering,
    median imputation, and standard scaling.
    """
    return Pipeline(
        steps=[
            (
                "feature_engineer",
                CreditRiskFeatureEngineer(
                    revolving_utilization_cap=revolving_utilization_cap,
                    age_min=age_min,
                    debt_ratio_cap=debt_ratio_cap,
                ),
            ),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

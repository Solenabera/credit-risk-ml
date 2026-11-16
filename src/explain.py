"""
Model Explainability and Adverse Action Reason Code Generator.
Computes SHAP feature attributions and maps them to regulatory adverse action notices.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline
from src.features import CreditRiskFeatureEngineer

# Regulatory Adverse Action Reason Mapping (FCRA / ECOA compliant descriptions)
REASON_CODE_MAPPING = {
    "RevolvingUtilizationOfUnsecuredLines": "High utilization or balance on revolving credit cards",
    "TotalPastDue": "History of past-due payments across credit accounts",
    "NumberOfTimes90DaysLate": "Serious delinquency: 90 or more days past due on credit accounts",
    "NumberOfTime30-59DaysPastDueNotWorse": "Recent 30 to 59 days late payment delinquency",
    "NumberOfTime60-89DaysPastDueNotWorse": "Recent 60 to 89 days late payment delinquency",
    "DebtRatio": "High total debt payment obligations relative to income",
    "MonthlyIncome": "Insufficient verifiable gross monthly income",
    "MonthlyIncome_Missing": "Missing or unverifiable monthly income information",
    "NumberOfOpenCreditLinesAndLoans": "Number of open credit lines and loans",
    "NumberRealEstateLoansOrLines": "Number or burden of real estate and mortgage loans",
    "age": "Limited length of credit history / age profile",
    "IncomePerCapita": "Low disposable income per dependent family member",
    "EstimatedDebtPayment": "High estimated monthly debt servicing burden",
    "DelinquencyDataAnomaly": "Discrepancies or anomalies detected in credit bureau delinquency records",
}


class CreditRiskExplainer:
    """
    Explainer class for credit scoring models using TreeExplainer on tree models
    or KernelExplainer / LinearExplainer.
    """

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.classifier = pipeline.named_steps["classifier"]
        
        # Unwrap calibrated classifier if present
        if hasattr(self.classifier, "base_estimator"):
            self.base_model = self.classifier.base_estimator
        elif hasattr(self.classifier, "calibrated_classifiers_"):
            # Use the base estimator from the first calibrated fold
            self.base_model = self.classifier.calibrated_classifiers_[0].estimator
        elif hasattr(self.classifier, "estimator"):
            self.base_model = self.classifier.estimator
        else:
            self.base_model = self.classifier

        self.explainer = shap.TreeExplainer(self.base_model)
        self.feature_names = self.preprocessor.named_steps["feature_engineer"].get_feature_names_out()

    def explain_instance(self, raw_input_df: pd.DataFrame, top_k: int = 4) -> Dict[str, Any]:
        """
        Explain a single applicant's decision and return top adverse action reason codes.
        """
        processed_arr = self.preprocessor.transform(raw_input_df)
        shap_values = self.explainer.shap_values(processed_arr)

        # Handle binary classification shap output format
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # Positive class (default)
        elif len(shap_values.shape) == 2:
            sv = shap_values[0]
        else:
            sv = shap_values

        contributions = []
        for name, val in zip(self.feature_names, sv):
            feat_name = str(name)
            contributions.append(
                {
                    "feature": feat_name,
                    "shap_value": round(float(val), 4),
                    "impact": "INCREASES_RISK" if val > 0 else "DECREASES_RISK",
                    "adverse_reason": REASON_CODE_MAPPING.get(feat_name, f"Factor related to {feat_name}"),
                }
            )

        # Sort by SHAP value descending (factors that increase risk most)
        risk_increasing = [c for c in contributions if c["shap_value"] > 0]
        risk_increasing.sort(key=lambda x: x["shap_value"], reverse=True)

        top_reasons = [c["adverse_reason"] for c in risk_increasing[:top_k]]
        if not top_reasons:
            top_reasons = ["Favorable overall credit profile with low default indicators"]

        return {
            "adverse_action_reasons": top_reasons,
            "feature_contributions": contributions,
        }

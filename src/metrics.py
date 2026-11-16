"""
Credit Risk and Financial Scoring Metrics Module.
Calculates industry-standard credit metrics: ROC-AUC, Gini, Kolmogorov-Smirnov (KS),
Brier Score, FICO-scale Credit Score, and Population Stability Index (PSI).
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
)


def calculate_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the Kolmogorov-Smirnov (KS) statistic and optimal KS threshold.
    KS measures the maximum vertical separation between cumulative good and bad distributions.
    
    Returns:
        (ks_statistic, optimal_threshold)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks_distances = tpr - fpr
    optimal_idx = np.argmax(ks_distances)
    ks_stat = float(ks_distances[optimal_idx])
    ks_threshold = float(thresholds[optimal_idx])
    return ks_stat, ks_threshold


def calculate_gini(roc_auc: float) -> float:
    """
    Calculate the Gini coefficient from ROC-AUC: Gini = 2 * AUC - 1.
    Standard metric required by financial regulatory frameworks (Basel II/III).
    """
    return float(2.0 * roc_auc - 1.0)


def pd_to_credit_score(
    pd_series: np.ndarray,
    base_score: float = 700.0,
    base_odds: float = 20.0,
    pdo: float = 40.0,
    min_score: int = 300,
    max_score: int = 850,
) -> np.ndarray:
    """
    Map calibrated Probability of Default (PD) to standard Credit Score (300-850)
    using the industry-standard Points to Double Odds (PDO) formula:
        Score = Offset + Factor * ln(Odds)
        where Odds = (1 - PD) / PD
        Factor = PDO / ln(2)
        Offset = BaseScore - Factor * ln(BaseOdds)
    """
    eps = 1e-7
    clamped_pd = np.clip(pd_series, eps, 1.0 - eps)
    odds = (1.0 - clamped_pd) / clamped_pd

    factor = pdo / np.log(2.0)
    offset = base_score - factor * np.log(base_odds)

    raw_scores = offset + factor * np.log(odds)
    scores = np.clip(np.round(raw_scores), min_score, max_score).astype(int)
    return scores


def assign_risk_tier(score: int, pd_prob: Optional[float] = None, threshold: Optional[float] = None) -> Dict[str, str]:
    """
    Assign credit risk tier and recommended decision based on credit score and default cutoff threshold.
    """
    if score >= 740:
        tier = "Tier 1 - Excellent"
        risk_level = "LOW"
        decision = "APPROVE_INSTANT"
    elif score >= 670:
        tier = "Tier 2 - Good"
        risk_level = "MODERATE"
        decision = "APPROVE_STANDARD"
    elif score >= 580:
        tier = "Tier 3 - Fair"
        risk_level = "ELEVATED"
        # If score is Fair but probability is below risk threshold, approve with review or standard
        decision = "APPROVE_STANDARD" if (pd_prob is not None and threshold is not None and pd_prob < threshold) else "MANUAL_REVIEW"
    else:
        tier = "Tier 4 - Subprime"
        risk_level = "HIGH"
        decision = "REJECT"

    return {"tier": tier, "decision": decision, "risk_level": risk_level}


def calculate_credit_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute comprehensive financial risk and classification metrics.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob)
    y_pred = (y_prob_arr >= threshold).astype(int)

    auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    gini = calculate_gini(auc)
    ks_stat, ks_thresh = calculate_ks_statistic(y_true_arr, y_prob_arr)
    brier = float(brier_score_loss(y_true_arr, y_prob_arr))

    precision = float(precision_score(y_true_arr, y_pred, zero_division=0))
    recall = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "roc_auc": round(auc, 4),
        "gini_coefficient": round(gini, 4),
        "ks_statistic": round(ks_stat, 4),
        "ks_optimal_threshold": round(ks_thresh, 4),
        "brier_score": round(brier, 4),
        "decision_threshold": round(threshold, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "specificity": round(specificity, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> Tuple[float, pd.DataFrame]:
    """
    Compute Population Stability Index (PSI) to detect distribution drift.
    
    Interpretation:
        PSI < 0.10: No significant distribution change (Stable)
        0.10 <= PSI < 0.25: Moderate shift; monitor closely
        PSI >= 0.25: Significant shift; model retraining recommended
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    # Use expected percentiles to define bucket edges
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bucket_edges = np.percentile(expected, percentiles)
    bucket_edges[0] = -np.inf
    bucket_edges[-1] = np.inf

    # Calculate counts in each bucket
    expected_counts, _ = np.histogram(expected, bins=bucket_edges)
    actual_counts, _ = np.histogram(actual, bins=bucket_edges)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Replace zero percentages with epsilon to prevent division by zero / log(0)
    expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
    actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)

    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    total_psi = float(np.sum(psi_values))

    bucket_df = pd.DataFrame(
        {
            "bucket": list(range(1, num_buckets + 1)),
            "expected_pct": expected_pct,
            "actual_pct": actual_pct,
            "psi_contribution": psi_values,
        }
    )

    return total_psi, bucket_df

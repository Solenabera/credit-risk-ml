"""
Unit Tests for Financial Scoring Metrics, Calibration, and PSI.
"""

import numpy as np
from src.metrics import (
    calculate_ks_statistic,
    calculate_gini,
    pd_to_credit_score,
    assign_risk_tier,
    calculate_credit_metrics,
    calculate_psi,
)


def test_calculate_gini():
    """Verify Gini = 2 * AUC - 1."""
    assert calculate_gini(0.50) == 0.0
    assert calculate_gini(1.00) == 1.0
    assert round(calculate_gini(0.85), 2) == 0.70


def test_calculate_ks_statistic():
    """Verify KS statistic correctly measures separation."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Perfect separation
    y_prob = np.array([0.05, 0.10, 0.15, 0.20, 0.80, 0.85, 0.90, 0.95])
    ks_stat, _ = calculate_ks_statistic(y_true, y_prob)
    assert ks_stat == 1.0


def test_pd_to_credit_score_range():
    """Verify credit score remains within 300 to 850 range."""
    # Extremely low PD
    low_pd = np.array([0.00001])
    score_low = pd_to_credit_score(low_pd)
    assert score_low[0] <= 850

    # Extremely high PD
    high_pd = np.array([0.99999])
    score_high = pd_to_credit_score(high_pd)
    assert score_high[0] >= 300
    assert score_high[0] < score_low[0]


def test_assign_risk_tier():
    """Verify tier mapping and decision policies."""
    tier1 = assign_risk_tier(780)
    assert tier1["tier"] == "Tier 1 - Excellent"
    assert tier1["decision"] == "APPROVE_INSTANT"

    tier4 = assign_risk_tier(450)
    assert tier4["tier"] == "Tier 4 - Subprime"
    assert tier4["decision"] == "REJECT"


def test_calculate_psi_no_drift():
    """Verify PSI between identical distributions is zero."""
    baseline = np.random.normal(100, 15, 1000)
    current = baseline.copy()

    psi, _ = calculate_psi(baseline, current)
    assert psi < 0.01


def test_calculate_psi_significant_drift():
    """Verify PSI detects significant distribution shift."""
    baseline = np.random.normal(100, 15, 1000)
    shifted = np.random.normal(150, 15, 1000)

    psi, _ = calculate_psi(baseline, shifted)
    assert psi > 0.25

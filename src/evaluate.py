"""
Holdout Test Set Evaluation Module.
Evaluates production model pipeline strictly on the untouched test partition.
"""

import json
import logging
from typing import Dict, Any, Optional
import joblib
import pandas as pd
from src.config import load_config, get_project_root
from src.data_loader import load_raw_data, get_stratified_split
from src.metrics import calculate_credit_metrics, pd_to_credit_score, assign_risk_tier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_holdout_test_set(
    pipeline_path: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Perform unbiased evaluation on the holdout test set.
    """
    config = load_config(config_path)
    root = get_project_root()

    if pipeline_path is None:
        pipeline_path = str(root / config["artifacts"]["pipeline_file"])

    metadata_path = root / config["artifacts"]["metadata_file"]
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    threshold = float(metadata.get("optimal_threshold", 0.0668))

    logger.info("Loading production pipeline from %s", pipeline_path)
    pipeline = joblib.load(pipeline_path)

    # Load data and retrieve the exact test partition
    df = load_raw_data()
    split_cfg = config["data"]["split"]
    _, _, _, _, X_test, y_test = get_stratified_split(
        df,
        target_col=config["data"]["target_column"],
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
        random_state=split_cfg["random_state"],
    )

    logger.info("Scoring holdout test set with %d records...", len(X_test))
    # Pipeline handles raw DataFrame -> features -> imputation -> scaling -> calibrated predict_proba
    y_test_prob = pipeline.predict_proba(X_test)[:, 1]

    # Calculate comprehensive metrics
    test_metrics = calculate_credit_metrics(y_test, y_test_prob, threshold=threshold)

    # Compute credit scores on test set
    credit_scores = pd_to_credit_score(
        y_test_prob,
        base_score=config["credit_score"]["base_score"],
        base_odds=config["credit_score"]["base_odds"],
        pdo=config["credit_score"]["pdo"],
    )

    # Tier distribution
    tier_counts = {}
    for score in credit_scores:
        t = assign_risk_tier(score)["tier"]
        tier_counts[t] = tier_counts.get(t, 0) + 1

    report = {
        "model_version": metadata.get("model_version", "1.0.0"),
        "test_sample_count": len(X_test),
        "test_default_count": int(y_test.sum()),
        "test_default_rate_pct": round(float(y_test.mean() * 100), 2),
        "decision_threshold_applied": threshold,
        "metrics": test_metrics,
        "credit_score_summary": {
            "mean": round(float(credit_scores.mean()), 1),
            "median": int(pd.Series(credit_scores).median()),
            "min": int(credit_scores.min()),
            "max": int(credit_scores.max()),
            "tier_distribution": tier_counts,
        },
    }

    report_path = root / config["artifacts"]["dir"] / "test_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Test Evaluation Complete:\n%s", json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    evaluate_holdout_test_set()

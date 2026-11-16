"""
End-to-End Pipeline Inference and Consistency Tests.
"""

import os
import joblib
import pandas as pd
from src.config import load_config, get_project_root


def test_production_pipeline_file_exists():
    """Verify serialized pipeline artifact exists on disk."""
    cfg = load_config()
    pipeline_path = get_project_root() / cfg["artifacts"]["pipeline_file"]
    assert os.path.exists(pipeline_path), f"Artifact missing at {pipeline_path}"


def test_production_pipeline_inference(sample_raw_dataframe):
    """Verify loaded pipeline runs inference and outputs calibrated probabilities."""
    cfg = load_config()
    pipeline_path = get_project_root() / cfg["artifacts"]["pipeline_file"]
    pipeline = joblib.load(pipeline_path)

    probs = pipeline.predict_proba(sample_raw_dataframe)
    assert probs.shape == (2, 2)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    # Check row probabilities sum to 1.0
    row_sums = probs.sum(axis=1)
    for s in row_sums:
        assert abs(s - 1.0) < 1e-5


def test_model_metadata_consistency():
    """Verify model metadata matches expectations."""
    cfg = load_config()
    meta_path = get_project_root() / cfg["artifacts"]["metadata_file"]
    assert os.path.exists(meta_path)

    import json
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert "model_version" in meta
    assert "optimal_threshold" in meta
    assert meta["validation_metrics"]["roc_auc"] > 0.80
    assert meta["validation_metrics"]["ks_statistic"] > 0.40

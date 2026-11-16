"""
Model Training and Experiment Tracking Pipeline.
Trains, tunes, calibrates, and packages the Credit Risk Scoring model with MLflow.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import xgboost as xgb

from src.calibration import CalibratedBinaryClassifier

from src.config import load_config, get_project_root
from src.data_loader import load_raw_data, get_stratified_split
from src.features import build_preprocessing_pipeline
from src.metrics import calculate_credit_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def find_cost_sensitive_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = 5000.0,
    cost_fp: float = 500.0,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Find threshold minimizing expected business loss.
    cost_fn: financial loss when a default occurs (False Negative, loan approved but defaulted).
    cost_fp: lost interest profit when a good borrower is rejected (False Positive).
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.50, 100)

    best_thresh = 0.10
    min_cost = float("inf")

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        # 1 = default, 0 = good
        # FN: y_true=1, y_pred=0 (predicted good, actually defaulted)
        fn = np.sum((y_true == 1) & (y_pred == 0))
        # FP: y_true=0, y_pred=1 (predicted default, actually good)
        fp = np.sum((y_true == 0) & (y_pred == 1))
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        if total_cost < min_cost:
            min_cost = total_cost
            best_thresh = float(t)

    return best_thresh, min_cost


def train_credit_scoring_system(config_path: str = None) -> Dict[str, Any]:
    """
    End-to-end model training, probability calibration, threshold selection,
    and MLflow experiment tracking.
    """
    config = load_config(config_path)
    root = get_project_root()

    # 1. Setup MLflow Tracking
    tracking_uri = config["mlflow"]["tracking_uri"]
    experiment_name = config["mlflow"]["experiment_name"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow configured. Tracking URI: %s, Experiment: %s", tracking_uri, experiment_name)

    # 2. Ingest and Split Data
    df = load_raw_data()
    split_cfg = config["data"]["split"]
    X_train, y_train, X_val, y_val, X_test, y_test = get_stratified_split(
        df,
        target_col=config["data"]["target_column"],
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
        random_state=split_cfg["random_state"],
    )

    # 3. Fit Preprocessing Pipeline ONLY on Train Set
    cap_cfg = config["features"]["capping"]
    preprocessor = build_preprocessing_pipeline(
        revolving_utilization_cap=cap_cfg["revolving_utilization_max"],
        age_min=cap_cfg["age_min"],
        debt_ratio_cap=cap_cfg["debt_ratio_max"],
    )

    logger.info("Fitting preprocessing pipeline on training partition...")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    feature_names = preprocessor.named_steps["feature_engineer"].get_feature_names_out()

    with mlflow.start_run(run_name=f"credit_scoring_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        run_id = run.info.run_id
        logger.info("Active MLflow Run ID: %s", run_id)

        # 4. Train Baseline Model (Logistic Regression)
        logger.info("Training Baseline Model (Logistic Regression)...")
        baseline_lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        baseline_lr.fit(X_train_proc, y_train)
        y_val_prob_base = baseline_lr.predict_proba(X_val_proc)[:, 1]
        baseline_metrics = calculate_credit_metrics(y_val, y_val_prob_base)
        logger.info("Baseline Val ROC-AUC: %.4f | KS: %.4f", baseline_metrics["roc_auc"], baseline_metrics["ks_statistic"])

        # 5. Train XGBoost Classifier with Early Stopping on Validation Set
        logger.info("Training Tuned XGBoost Classifier...")
        xgb_params = config["model"]["params"].copy()
        
        # Calculate imbalance weight
        neg_count = int((y_train == 0).sum())
        pos_count = int((y_train == 1).sum())
        imbalance_ratio = neg_count / pos_count
        logger.info("Train class balance: Negatives=%d, Positives=%d (Ratio: %.2f:1)", neg_count, pos_count, imbalance_ratio)

        xgb_model = xgb.XGBClassifier(
            **xgb_params,
            early_stopping_rounds=20,
        )
        xgb_model.fit(
            X_train_proc,
            y_train,
            eval_set=[(X_val_proc, y_val)],
            verbose=False,
        )

        # 6. Calibrate Model Probabilities
        # Ensures predict_proba matches empirical Probability of Default
        calib_cfg = config["model"]["calibration"]
        if calib_cfg.get("enabled", True):
            calib_method = calib_cfg.get("method", "isotonic")
            logger.info("Applying %s probability calibration using validation set...", calib_method)
            calibrated_classifier = CalibratedBinaryClassifier(
                base_estimator=xgb_model,
                method=calib_method,
            )
            calibrated_classifier.fit_calibrator(X_val_proc, y_val)
            final_estimator = calibrated_classifier
        else:
            final_estimator = xgb_model

        # 7. Validation Evaluation & Threshold Optimization
        y_val_prob = final_estimator.predict_proba(X_val_proc)[:, 1]
        
        # Determine KS-optimal and Cost-sensitive thresholds
        ks_stat, ks_threshold = calculate_credit_metrics(y_val, y_val_prob)["ks_statistic"], calculate_credit_metrics(y_val, y_val_prob)["ks_optimal_threshold"]
        cost_threshold, _ = find_cost_sensitive_threshold(y_val.to_numpy(), y_val_prob)
        
        # We select cost-sensitive threshold or KS threshold as optimal operational threshold
        selected_threshold = float(ks_threshold)
        val_metrics = calculate_credit_metrics(y_val, y_val_prob, threshold=selected_threshold)
        logger.info("Calibrated Val Metrics at threshold %.4f: ROC-AUC=%.4f, KS=%.4f, Gini=%.4f, Brier=%.4f",
                    selected_threshold, val_metrics["roc_auc"], val_metrics["ks_statistic"],
                    val_metrics["gini_coefficient"], val_metrics["brier_score"])

        # 8. Assemble Production Pipeline
        production_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", final_estimator),
            ]
        )

        # 9. Log Parameters and Metrics to MLflow
        mlflow.log_params(xgb_params)
        mlflow.log_param("calibration_enabled", calib_cfg.get("enabled", True))
        mlflow.log_param("calibration_method", calib_cfg.get("method", "isotonic"))
        mlflow.log_param("optimal_threshold", selected_threshold)
        mlflow.log_param("cost_threshold", cost_threshold)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))})
        mlflow.log_metrics({"baseline_val_roc_auc": baseline_metrics["roc_auc"], "baseline_val_ks": baseline_metrics["ks_statistic"]})

        # 10. Serialize Production Artifacts
        artifacts_dir = root / config["artifacts"]["dir"]
        os.makedirs(artifacts_dir, exist_ok=True)

        pipeline_path = root / config["artifacts"]["pipeline_file"]
        joblib.dump(production_pipeline, pipeline_path)
        logger.info("Saved unified production pipeline to %s", pipeline_path)

        threshold_path = root / config["artifacts"]["threshold_file"]
        with open(threshold_path, "w", encoding="utf-8") as f:
            f.write(str(selected_threshold))

        # Save comprehensive metadata
        metadata = {
            "model_name": config["project"]["name"],
            "model_version": config["project"]["version"],
            "training_timestamp": datetime.now(timezone.utc).isoformat(),
            "mlflow_run_id": run_id,
            "algorithm": "XGBoost + CalibratedBinaryClassifier (Isotonic)",
            "num_features": len(feature_names),
            "feature_names": list(feature_names),
            "optimal_threshold": selected_threshold,
            "cost_sensitive_threshold": cost_threshold,
            "validation_metrics": val_metrics,
            "credit_score_config": config["credit_score"],
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "test_samples": len(X_test),
        }
        metadata_path = root / config["artifacts"]["metadata_file"]
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved model metadata to %s", metadata_path)

        # 11. Log Model to MLflow Model Registry
        mlflow.sklearn.log_model(
            sk_model=production_pipeline,
            artifact_path="credit_scoring_pipeline",
            registered_model_name="CreditRiskScoringPipeline",
            serialization_format="cloudpickle",
        )

        return {
            "run_id": run_id,
            "pipeline_path": str(pipeline_path),
            "metadata_path": str(metadata_path),
            "val_metrics": val_metrics,
            "optimal_threshold": selected_threshold,
        }


if __name__ == "__main__":
    train_credit_scoring_system()

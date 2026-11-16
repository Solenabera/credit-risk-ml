"""
Enterprise Credit Risk Scoring API.
FastAPI microservice providing real-time loan underwriting predictions,
FICO-style credit scores, risk tier classification, and FCRA/ECOA adverse action explanations.
"""

import os
import sys
import time
import json
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure root is on path for modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.config import settings
from api.schemas import (
    LoanApplication,
    BatchLoanApplicationRequest,
    PredictionResponse,
    BatchPredictionResponse,
    ExplanationResponse,
    HealthResponse,
    ModelInfoResponse,
    RiskDecisionEnum,
    FeatureContribution,
)
from src.config import load_config, get_project_root
from src.metrics import pd_to_credit_score, assign_risk_tier
from src.explain import CreditRiskExplainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("credit_scoring_api")

# Application State
app_state: Dict[str, Any] = {
    "pipeline": None,
    "explainer": None,
    "metadata": {},
    "optimal_threshold": 0.0742,
    "start_time": time.time(),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application Lifespan Context Manager.
    Loads models, metadata, and explainers at startup and cleans up on shutdown.
    """
    root = get_project_root()
    cfg = load_config()
    pipeline_path = root / cfg["artifacts"]["pipeline_file"]
    metadata_path = root / cfg["artifacts"]["metadata_file"]
    threshold_path = root / cfg["artifacts"]["threshold_file"]

    logger.info("Initializing Credit Risk Scoring microservice...")
    try:
        if os.path.exists(pipeline_path):
            logger.info("Loading ML Pipeline from %s", pipeline_path)
            app_state["pipeline"] = joblib.load(pipeline_path)
            logger.info("Model pipeline loaded successfully.")

            # Initialize SHAP explainer
            logger.info("Initializing SHAP TreeExplainer for Adverse Action generation...")
            app_state["explainer"] = CreditRiskExplainer(app_state["pipeline"])
            logger.info("SHAP explainer ready.")
        else:
            logger.warning("Pipeline artifact not found at %s. Service in degraded state.", pipeline_path)

        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                app_state["metadata"] = json.load(f)
            app_state["optimal_threshold"] = float(app_state["metadata"].get("optimal_threshold", 0.0742))
        elif os.path.exists(threshold_path):
            with open(threshold_path, "r", encoding="utf-8") as f:
                app_state["optimal_threshold"] = float(f.read().strip())

        logger.info("Operating with optimal decision threshold: %.4f", app_state["optimal_threshold"])

    except Exception as exc:
        logger.exception("Critical error during application startup: %s", exc)
        raise RuntimeError(f"Service initialization failed: {exc}") from exc

    app_state["start_time"] = time.time()
    yield

    # Teardown logic
    logger.info("Shutting down Credit Risk Scoring microservice...")
    app_state["pipeline"] = None
    app_state["explainer"] = None


# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add timing header to all outgoing responses."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(duration_ms)
    return response


def _score_application(app_data: Dict[str, Any], include_shap: bool = True) -> PredictionResponse:
    """Internal helper to score a single application dictionary."""
    pipeline = app_state["pipeline"]
    threshold = app_state["optimal_threshold"]

    # Extract ID
    app_id = app_data.pop("application_id", None)

    # Convert to single-row DataFrame
    input_df = pd.DataFrame([app_data])

    # Run inference through Scikit-Learn unified pipeline
    probabilities = pipeline.predict_proba(input_df)
    pd_prob = float(probabilities[0][1])

    # Map to FICO-scale Credit Score (300 to 850)
    scores = pd_to_credit_score(pd_prob)
    score_val = int(scores) if isinstance(scores, (int, float, pd.Series)) else int(scores.item())

    tier_info = assign_risk_tier(score_val, pd_prob=pd_prob, threshold=threshold)
    decision = RiskDecisionEnum(tier_info["decision"])

    adverse_reasons = []
    if include_shap and app_state["explainer"] is not None:
        try:
            explanation = app_state["explainer"].explain_instance(input_df, top_k=4)
            adverse_reasons = explanation["adverse_action_reasons"]
        except Exception as e:
            logger.warning("SHAP explanation generation failed: %s", e)

    return PredictionResponse(
        application_id=app_id,
        default_probability=round(pd_prob, 4),
        credit_score=score_val,
        risk_tier=tier_info["tier"],
        risk_decision=decision,
        decision_threshold=round(threshold, 4),
        adverse_action_reasons=adverse_reasons,
    )


@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint with service information and links to interactive API documentation."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "online",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Liveness probe reporting model status and service uptime."""
    is_loaded = app_state["pipeline"] is not None
    uptime = time.time() - app_state["start_time"]
    version = app_state["metadata"].get("model_version", settings.app_version)
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_version=version,
        uptime_seconds=round(uptime, 1),
    )


@app.get("/health/ready", tags=["Monitoring"])
async def readiness_check():
    """Readiness probe used by Kubernetes / container orchestrators."""
    if app_state["pipeline"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded into memory.",
        )
    return {"ready": True}


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model Governance"])
async def model_info():
    """Governance endpoint returning model version, algorithm, training metadata, and metrics."""
    meta = app_state["metadata"]
    if not meta:
        raise HTTPException(status_code=404, detail="Model metadata is unavailable.")

    return ModelInfoResponse(
        model_name=meta.get("model_name", settings.app_name),
        model_version=meta.get("model_version", settings.app_version),
        algorithm=meta.get("algorithm", "Unknown"),
        training_timestamp=meta.get("training_timestamp", "Unknown"),
        features=meta.get("feature_names", []),
        optimal_threshold=float(meta.get("optimal_threshold", 0.0742)),
        validation_metrics=meta.get("validation_metrics", {}),
        credit_score_config=meta.get("credit_score_config", {}),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_credit_risk(application: LoanApplication):
    """
    Score a single loan application.
    Returns calibrated Probability of Default (PD), FICO-equivalent credit score,
    risk tier, automated decision, and FCRA/ECOA Adverse Action reason codes.
    """
    if app_state["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")

    try:
        app_dict = application.model_dump(by_alias=True)
        return _score_application(app_dict, include_shap=True)
    except Exception as exc:
        logger.exception("Error scoring loan application: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}") from exc


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_credit_risk_batch(batch_req: BatchLoanApplicationRequest):
    """
    Perform high-throughput batch scoring on multiple loan applications.
    """
    if app_state["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")

    predictions = []
    approved = 0
    review = 0
    rejected = 0

    for app_item in batch_req.applications:
        app_dict = app_item.model_dump(by_alias=True)
        pred = _score_application(app_dict, include_shap=False)
        predictions.append(pred)

        if pred.risk_decision in [RiskDecisionEnum.APPROVE_INSTANT, RiskDecisionEnum.APPROVE_STANDARD]:
            approved += 1
        elif pred.risk_decision == RiskDecisionEnum.MANUAL_REVIEW:
            review += 1
        else:
            rejected += 1

    return BatchPredictionResponse(
        total_processed=len(predictions),
        approved_count=approved,
        review_count=review,
        rejected_count=rejected,
        predictions=predictions,
    )


@app.post("/explain", response_model=ExplanationResponse, tags=["Explainability"])
async def explain_credit_decision(application: LoanApplication):
    """
    Generate detailed SHAP feature attributions and Adverse Action reason codes
    for a specific credit decision.
    """
    if app_state["pipeline"] is None or app_state["explainer"] is None:
        raise HTTPException(status_code=503, detail="Model explainer is not loaded.")

    try:
        app_dict = application.model_dump(by_alias=True)
        app_id = app_dict.pop("application_id", None)
        input_df = pd.DataFrame([app_dict])

        # Get probabilities and score
        probs = app_state["pipeline"].predict_proba(input_df)
        pd_prob = float(probs[0][1])
        score = int(pd_to_credit_score(pd_prob))

        # Get SHAP explanation
        explanation = app_state["explainer"].explain_instance(input_df, top_k=4)

        contributions = [
            FeatureContribution(
                feature=c["feature"],
                shap_value=c["shap_value"],
                impact=c["impact"],
                adverse_reason=c["adverse_reason"],
            )
            for c in explanation["feature_contributions"]
        ]

        return ExplanationResponse(
            application_id=app_id,
            default_probability=round(pd_prob, 4),
            credit_score=score,
            adverse_action_reasons=explanation["adverse_action_reasons"],
            feature_contributions=contributions,
        )

    except Exception as exc:
        logger.exception("Error generating SHAP explanation: %s", exc)
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(exc)}") from exc

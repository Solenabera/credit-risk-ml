import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from api.schemas import LoanApplication, PredictionResponse
import sys
import os

# Ensure the app can find the src module for the custom transformer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features import CreditRiskFeatureEngineer

# Global variables to hold our model and threshold
ml_pipeline = None
optimal_threshold = 0.5 

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Executes on application startup. 
    Loads the heavy ML artifacts into memory so they are ready for instant inference.
    """
    global ml_pipeline, optimal_threshold
    
    pipeline_path = os.path.join("artifacts", "credit_scoring_pipeline_v1.joblib")
    threshold_path = os.path.join("artifacts", "optimal_threshold.txt")
    
    try:
        print(f"Loading ML Pipeline from {pipeline_path}...")
        ml_pipeline = joblib.load(pipeline_path)
        
        if os.path.exists(threshold_path):
            with open(threshold_path, "r") as f:
                optimal_threshold = float(f.read().strip())
        print(f"Model loaded successfully. Operating with threshold: {optimal_threshold:.4f}")
    except Exception as e:
        print(f"Critical Error loading model: {e}")
        raise RuntimeError("Failed to load ML artifacts. Check paths.")
        
    yield # The application runs while yielding
    
    # Cleanup logic (if any) would go here on shutdown
    ml_pipeline = None

# Initialize the FastAPI app
app = FastAPI(
    title="Credit Risk Scoring API",
    description="Enterprise API for predicting loan default probabilities.",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/predict", response_model=PredictionResponse)
async def predict_risk(application: LoanApplication):
    """
    Receives a loan application, runs it through the XGBoost pipeline, 
    and returns the probability of default.
    """
    if ml_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        # Convert the validated Pydantic object to a pandas DataFrame
        # by_alias=True ensures the columns match what the model expects
        input_data = pd.DataFrame([application.model_dump(by_alias=True)])
        
        # Run inference using the loaded pipeline
        # The pipeline automatically handles the custom feature engineering, imputation, and scaling
        probabilities = ml_pipeline.predict_proba(input_data)
        default_prob = float(probabilities[0][1])
        
        # Apply the business logic threshold we calculated in Phase 5
        decision = "REJECT" if default_prob >= optimal_threshold else "APPROVE"
        
        return PredictionResponse(
            default_probability=default_prob,
            risk_decision=decision
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/health")
async def health_check():
    """Simple endpoint to verify the API is running (used by Kubernetes/Docker)."""
    status = "healthy" if ml_pipeline is not None else "degraded"
    return {"status": status, "model_loaded": ml_pipeline is not None}
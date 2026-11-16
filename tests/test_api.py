from fastapi.testclient import TestClient
from api.main import app

def test_health_check():
    """Verify the API boots up and the ML model loads successfully."""
    # Using 'with' triggers the FastAPI lifespan event, loading the ML pipeline
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["model_loaded"] is True

def test_predict_endpoint_valid_payload():
    """Verify the API returns a valid prediction for a correct payload."""
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.5,
        "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.3,
        "MonthlyIncome": 6000.0,
        "NumberOfOpenCreditLinesAndLoans": 5,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2
    }
    
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        
        data = response.json()
        assert "default_probability" in data
        assert "risk_decision" in data
        assert 0.0 <= data["default_probability"] <= 1.0
        assert data["risk_decision"] in ["APPROVE", "REJECT"]

def test_predict_endpoint_invalid_payload():
    """Verify Pydantic blocks invalid data (e.g., negative age)."""
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.5,
        "age": -5, 
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.3,
        "NumberOfOpenCreditLinesAndLoans": 5,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0
    }
    
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
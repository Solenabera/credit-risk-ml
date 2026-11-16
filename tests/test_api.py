"""
Integration and Unit Tests for FastAPI Serving Microservice.
"""

from api.schemas import RiskDecisionEnum


def test_root_endpoint(client):
    """Verify service root returns 200 and docs link."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "/docs" in data["documentation"]


def test_health_endpoint(client):
    """Verify /health returns 200, healthy status, and loaded model."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "uptime_seconds" in data


def test_readiness_endpoint(client):
    """Verify /health/ready returns 200."""
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_model_info_endpoint(client):
    """Verify /model/info returns governance metadata, features, and metrics."""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "model_version" in data
    assert "validation_metrics" in data
    assert len(data["features"]) > 0
    assert 0.0 < data["optimal_threshold"] < 1.0


def test_predict_prime_applicant(client, prime_applicant_payload):
    """Verify prime applicant receives low default probability and approval."""
    response = client.post("/predict", json=prime_applicant_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["application_id"] == "APP-PRIME-001"
    assert 0.0 <= data["default_probability"] <= 0.10
    assert data["credit_score"] >= 700
    assert data["risk_decision"] in [
        RiskDecisionEnum.APPROVE_INSTANT.value,
        RiskDecisionEnum.APPROVE_STANDARD.value,
    ]
    assert 300 <= data["credit_score"] <= 850


def test_predict_subprime_applicant(client, subprime_applicant_payload):
    """Verify subprime applicant receives high probability, subprime score, and REJECT."""
    response = client.post("/predict", json=subprime_applicant_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["application_id"] == "APP-SUBPRIME-999"
    assert data["default_probability"] > 0.50
    assert data["credit_score"] < 580
    assert data["risk_decision"] == RiskDecisionEnum.REJECT.value
    assert len(data["adverse_action_reasons"]) > 0


def test_predict_validation_error_age(client, prime_applicant_payload):
    """Verify Pydantic blocks underage applicant (age < 18) with 422 Unprocessable Entity."""
    payload = prime_applicant_payload.copy()
    payload["age"] = 16
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_validation_error_negative_utilization(client, prime_applicant_payload):
    """Verify Pydantic blocks negative revolving credit utilization with 422."""
    payload = prime_applicant_payload.copy()
    payload["RevolvingUtilizationOfUnsecuredLines"] = -0.5
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_mandatory_field(client):
    """Verify omission of required field (DebtRatio) triggers 422 error."""
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.5,
        "age": 40,
        "NumberOfOpenCreditLinesAndLoans": 5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_batch_predict_endpoint(client, prime_applicant_payload, subprime_applicant_payload):
    """Verify batch inference handles multiple records and returns aggregated totals."""
    batch_payload = {
        "applications": [prime_applicant_payload, subprime_applicant_payload]
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_processed"] == 2
    assert data["approved_count"] >= 1
    assert data["rejected_count"] >= 1
    assert len(data["predictions"]) == 2


def test_explain_endpoint(client, subprime_applicant_payload):
    """Verify /explain returns SHAP feature contributions and reason codes."""
    response = client.post("/explain", json=subprime_applicant_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["application_id"] == "APP-SUBPRIME-999"
    assert len(data["adverse_action_reasons"]) > 0
    assert len(data["feature_contributions"]) > 0

    first_contrib = data["feature_contributions"][0]
    assert "feature" in first_contrib
    assert "shap_value" in first_contrib
    assert "impact" in first_contrib
    assert "adverse_reason" in first_contrib

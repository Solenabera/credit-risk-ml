"""
Pytest Fixtures for Credit Risk Scoring Test Suite.
"""

import pytest
import pandas as pd
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture(scope="session")
def client():
    """Session-scoped FastAPI TestClient running within lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def prime_applicant_payload():
    """High-quality prime applicant with low risk indicators."""
    return {
        "application_id": "APP-PRIME-001",
        "RevolvingUtilizationOfUnsecuredLines": 0.15,
        "age": 48,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.22,
        "MonthlyIncome": 9500.0,
        "NumberOfOpenCreditLinesAndLoans": 9,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 2,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1,
    }


@pytest.fixture
def subprime_applicant_payload():
    """High-risk applicant with past delinquency and severe utilization."""
    return {
        "application_id": "APP-SUBPRIME-999",
        "RevolvingUtilizationOfUnsecuredLines": 1.95,
        "age": 23,
        "NumberOfTime30-59DaysPastDueNotWorse": 3,
        "DebtRatio": 0.85,
        "MonthlyIncome": 1800.0,
        "NumberOfOpenCreditLinesAndLoans": 14,
        "NumberOfTimes90DaysLate": 2,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 1,
        "NumberOfDependents": 3,
    }


@pytest.fixture
def sample_raw_dataframe():
    """Sample DataFrame with raw features including edge cases and nulls."""
    return pd.DataFrame(
        [
            {
                "RevolvingUtilizationOfUnsecuredLines": 5.5,  # Needs capping
                "age": 14,                                  # Needs capping to 18
                "NumberOfTime30-59DaysPastDueNotWorse": 98, # Anomaly code
                "DebtRatio": 0.35,
                "MonthlyIncome": None,                       # Missing
                "NumberOfOpenCreditLinesAndLoans": 4,
                "NumberOfTimes90DaysLate": 96,              # Anomaly code
                "NumberRealEstateLoansOrLines": 0,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfDependents": None,                  # Missing
            },
            {
                "RevolvingUtilizationOfUnsecuredLines": 0.2,
                "age": 55,
                "NumberOfTime30-59DaysPastDueNotWorse": 0,
                "DebtRatio": 0.15,
                "MonthlyIncome": 12000.0,
                "NumberOfOpenCreditLinesAndLoans": 8,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 2,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 2,
            },
        ]
    )

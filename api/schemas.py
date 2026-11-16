"""
Pydantic Schemas for Credit Risk Scoring API.
Defines strict input request validation and structured response models.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class RiskDecisionEnum(str, Enum):
    APPROVE_INSTANT = "APPROVE_INSTANT"
    APPROVE_STANDARD = "APPROVE_STANDARD"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    REJECT = "REJECT"


class LoanApplication(BaseModel):
    """
    Loan application payload representing applicant credit bureau and demographic features.
    """
    application_id: Optional[str] = Field(None, description="Optional external loan application identifier")
    RevolvingUtilizationOfUnsecuredLines: float = Field(
        ...,
        ge=0.0,
        description="Total balance on credit cards and personal lines of credit divided by sum of credit limits",
        json_schema_extra={"example": 0.35},
    )
    age: int = Field(
        ...,
        ge=18,
        le=120,
        description="Age of borrower in years",
        json_schema_extra={"example": 42},
    )
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        0,
        ge=0,
        alias="NumberOfTime30-59DaysPastDueNotWorse",
        description="Number of times borrower has been 30-59 days past due in the past 2 years",
        json_schema_extra={"example": 0},
    )
    DebtRatio: float = Field(
        ...,
        ge=0.0,
        description="Monthly debt payments, alimony, living costs divided by monthly gross income",
        json_schema_extra={"example": 0.28},
    )
    MonthlyIncome: Optional[float] = Field(
        None,
        ge=0.0,
        description="Gross monthly income in USD (can be omitted/null if unverified)",
        json_schema_extra={"example": 6500.0},
    )
    NumberOfOpenCreditLinesAndLoans: int = Field(
        ...,
        ge=0,
        description="Number of open loans (installment, auto, mortgage) and lines of credit (credit cards)",
        json_schema_extra={"example": 8},
    )
    NumberOfTimes90DaysLate: int = Field(
        0,
        ge=0,
        description="Number of times borrower has been 90 or more days past due",
        json_schema_extra={"example": 0},
    )
    NumberRealEstateLoansOrLines: int = Field(
        0,
        ge=0,
        description="Number of mortgage and real estate loans including home equity lines of credit",
        json_schema_extra={"example": 1},
    )
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        0,
        ge=0,
        alias="NumberOfTime60-89DaysPastDueNotWorse",
        description="Number of times borrower has been 60-89 days past due in the past 2 years",
        json_schema_extra={"example": 0},
    )
    NumberOfDependents: Optional[int] = Field(
        0,
        ge=0,
        le=20,
        description="Number of dependents in family excluding themselves",
        json_schema_extra={"example": 2},
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "application_id": "APP-2026-001",
                "RevolvingUtilizationOfUnsecuredLines": 0.25,
                "age": 38,
                "NumberOfTime30-59DaysPastDueNotWorse": 0,
                "DebtRatio": 0.32,
                "MonthlyIncome": 7200.0,
                "NumberOfOpenCreditLinesAndLoans": 6,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 1,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 1,
            }
        },
    )


class BatchLoanApplicationRequest(BaseModel):
    """Batch request container for bulk credit decisioning."""
    applications: List[LoanApplication] = Field(..., min_length=1, max_length=1000)


class FeatureContribution(BaseModel):
    """Local feature attribution from SHAP explainability."""
    feature: str
    shap_value: float
    impact: str
    adverse_reason: str


class PredictionResponse(BaseModel):
    """Comprehensive credit scoring response."""
    application_id: Optional[str] = None
    default_probability: float = Field(..., description="Calibrated Probability of Default (PD) between 0.0 and 1.0")
    credit_score: int = Field(..., description="FICO-equivalent credit score between 300 and 850")
    risk_tier: str = Field(..., description="Credit risk rating tier")
    risk_decision: RiskDecisionEnum = Field(..., description="Automated business policy recommendation")
    decision_threshold: float = Field(..., description="Operational default cutoff threshold")
    adverse_action_reasons: List[str] = Field(
        default_factory=list,
        description="Principal reason codes required for FCRA/ECOA Adverse Action notices",
    )


class BatchPredictionResponse(BaseModel):
    """Batch evaluation response summary."""
    total_processed: int
    approved_count: int
    review_count: int
    rejected_count: int
    predictions: List[PredictionResponse]


class ExplanationResponse(BaseModel):
    """Detailed SHAP explainability response."""
    application_id: Optional[str] = None
    default_probability: float
    credit_score: int
    adverse_action_reasons: List[str]
    feature_contributions: List[FeatureContribution]


class HealthResponse(BaseModel):
    """Liveness and readiness health check response."""
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    """Model governance and metadata response."""
    model_name: str
    model_version: str
    algorithm: str
    training_timestamp: str
    features: List[str]
    optimal_threshold: float
    validation_metrics: Dict[str, Any]
    credit_score_config: Dict[str, Any]

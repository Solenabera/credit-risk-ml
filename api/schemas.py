from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

class LoanApplication(BaseModel):
    """
    Defines the exact structure and data types required for the API request.
    If a request does not match this schema, FastAPI automatically rejects it with a 422 error.
    """
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., ge=0, description="Total balance on credit cards and personal lines divided by the sum of credit limits")
    age: int = Field(..., ge=18, description="Age of borrower in years")
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(0, alias="NumberOfTime30-59DaysPastDueNotWorse")
    DebtRatio: float = Field(..., ge=0, description="Monthly debt payments divided by monthly gross income")
    MonthlyIncome: Optional[float] = Field(None, description="Monthly income (can be null)")
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0)
    NumberOfTimes90DaysLate: int = Field(0)
    NumberRealEstateLoansOrLines: int = Field(0)
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(0, alias="NumberOfTime60-89DaysPastDueNotWorse")
    NumberOfDependents: Optional[int] = Field(0)

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "RevolvingUtilizationOfUnsecuredLines": 0.766,
                "age": 45,
                "NumberOfTime30-59DaysPastDueNotWorse": 2,
                "DebtRatio": 0.8,
                "MonthlyIncome": 5000.0,
                "NumberOfOpenCreditLinesAndLoans": 13,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 6,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 2
            }
        }
    )

class PredictionResponse(BaseModel):
    """Defines the structure of the API response."""
    default_probability: float
    risk_decision: str
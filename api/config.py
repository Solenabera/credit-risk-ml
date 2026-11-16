"""
API Server Settings and Environment Configuration.
"""

import os
from typing import List


class ApiSettings:
    """Settings class for FastAPI application."""

    def __init__(self):
        self.app_name: str = "Credit Risk Scoring and Decision API"
        self.app_version: str = "1.0.0"
        self.app_description: str = (
            "Enterprise-grade Credit Scoring microservice with calibrated default probabilities, "
            "FICO-scale scoring, regulatory adverse action explanations, and drift monitoring."
        )
        self.env: str = os.getenv("APP_ENV", "production")
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"
        self.host: str = os.getenv("APP_HOST", "0.0.0.0")
        self.port: int = int(os.getenv("APP_PORT", "8000"))
        self.cors_origins: List[str] = [
            origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",")
        ]


settings = ApiSettings()

"""
Unit Tests for Custom Feature Engineering Transformers.
"""

import numpy as np
import pandas as pd
from src.features import CreditRiskFeatureEngineer, build_preprocessing_pipeline


def test_feature_engineer_capping(sample_raw_dataframe):
    """Verify capping of revolving utilization and biological age floor."""
    transformer = CreditRiskFeatureEngineer(revolving_utilization_cap=2.0, age_min=18)
    transformed = transformer.fit_transform(sample_raw_dataframe)

    # First row had utilization 5.5, should be capped to 2.0
    assert transformed["RevolvingUtilizationOfUnsecuredLines"].iloc[0] == 2.0
    # First row had age 14, should be capped to 18
    assert transformed["age"].iloc[0] == 18

    # Second row was within bounds
    assert transformed["RevolvingUtilizationOfUnsecuredLines"].iloc[1] == 0.2
    assert transformed["age"].iloc[1] == 55


def test_feature_engineer_anomaly_codes(sample_raw_dataframe):
    """Verify delinquency codes 96 and 98 are sanitized and anomaly flag is raised."""
    transformer = CreditRiskFeatureEngineer(delinquency_anomaly_codes=[96, 98])
    transformed = transformer.fit_transform(sample_raw_dataframe)

    # First row had 98 in 30-59 days and 96 in 90 days late
    assert transformed["NumberOfTime30-59DaysPastDueNotWorse"].iloc[0] == 0
    assert transformed["NumberOfTimes90DaysLate"].iloc[0] == 0
    assert transformed["DelinquencyDataAnomaly"].iloc[0] == 1.0

    # Second row had no anomaly codes
    assert transformed["DelinquencyDataAnomaly"].iloc[1] == 0.0


def test_feature_engineer_missing_indicators(sample_raw_dataframe):
    """Verify missing flags are created for MonthlyIncome and NumberOfDependents."""
    transformer = CreditRiskFeatureEngineer()
    transformed = transformer.fit_transform(sample_raw_dataframe)

    # First row had nulls
    assert transformed["MonthlyIncome_Missing"].iloc[0] == 1.0
    assert transformed["NumberOfDependents_Missing"].iloc[0] == 1.0

    # Second row had values
    assert transformed["MonthlyIncome_Missing"].iloc[1] == 0.0
    assert transformed["NumberOfDependents_Missing"].iloc[1] == 0.0


def test_feature_names_out(sample_raw_dataframe):
    """Verify get_feature_names_out returns expected list of transformed features."""
    transformer = CreditRiskFeatureEngineer()
    transformer.fit(sample_raw_dataframe)
    names = transformer.get_feature_names_out()

    assert len(names) == 17
    assert "TotalPastDue" in names
    assert "DelinquencyDataAnomaly" in names
    assert "MonthlyIncome_Missing" in names


def test_full_preprocessing_pipeline_fit_transform(sample_raw_dataframe):
    """Verify unified preprocessing pipeline transforms without error and handles nulls."""
    pipeline = build_preprocessing_pipeline()
    processed_arr = pipeline.fit_transform(sample_raw_dataframe)

    assert isinstance(processed_arr, np.ndarray)
    assert processed_arr.shape == (2, 17)
    # Check that SimpleImputer successfully filled all NaNs
    assert not np.isnan(processed_arr).any()

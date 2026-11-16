"""
Data Ingestion and Validation Module.
Handles loading raw credit datasets, validating schemas, and stratified splitting.
"""

from typing import Tuple, Optional, Dict, Any
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import load_config, get_project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]


def load_raw_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw credit risk training data and clean export artifacts.
    """
    if file_path is None:
        cfg = load_config()
        file_path = str(get_project_root() / cfg["data"]["raw_train_path"])

    logger.info("Loading raw dataset from %s", file_path)
    df = pd.read_csv(file_path)

    # Drop index column left over from database/CSV export
    for drop_col in ["Unnamed: 0", "Unnamed: 0.1"]:
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])

    logger.info("Loaded dataset with shape %s", df.shape)
    return df


def validate_input_schema(df: pd.DataFrame, is_inference: bool = False) -> bool:
    """
    Validate that the DataFrame contains expected columns and appropriate data types.
    """
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing mandatory columns: {missing}")

    if not is_inference and "SeriousDlqin2yrs" not in df.columns:
        raise ValueError("Training dataset must contain target column 'SeriousDlqin2yrs'.")

    return True


def get_stratified_split(
    df: pd.DataFrame,
    target_col: str = "SeriousDlqin2yrs",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split dataset into Train (70%), Validation (15%), and Test (15%) partitions
    stratified by the target to prevent data leakage and preserve class distribution.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
        raise ValueError("Ratios train + val + test must sum to 1.0.")

    validate_input_schema(df, is_inference=False)

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # First split off the final holdout test set
    test_size_first = test_ratio
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size_first, stratify=y, random_state=random_state
    )

    # Next split the remainder into Train and Validation
    val_size_second = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_second, stratify=y_temp, random_state=random_state
    )

    logger.info(
        "Dataset split complete: Train=%s (default rate=%.3f%%), Val=%s (default rate=%.3f%%), Test=%s (default rate=%.3f%%)",
        X_train.shape,
        y_train.mean() * 100,
        X_val.shape,
        y_val.mean() * 100,
        X_test.shape,
        y_test.mean() * 100,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test

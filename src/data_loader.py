"""
Data loading, validation, and leakage column removal.
"""

import os
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.utils import setup_logger

logger = setup_logger("data_loader")

EXPECTED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def load_raw_data(
    filepath: str,
    identifier_cols: List[str],
    leakage_cols: List[str],
    target_col: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load raw AI4I 2020 dataset, validate structure, and strip identifier and leakage columns.

    Args:
        filepath: Path to raw CSV file.
        identifier_cols: Columns used solely as IDs (UDI, Product ID).
        leakage_cols: Columns providing direct failure hints (TWF, HDF, PWF, OSF, RNF).
        target_col: Name of target column ('Machine failure').

    Returns:
        Tuple of:
          - Cleaned DataFrame containing only genuine predictors and target.
          - Dictionary containing dataset summary metadata.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw dataset file not found at: {filepath}")

    logger.info(f"Loading raw dataset from {filepath}")
    df_raw = pd.read_csv(filepath)

    # Validate presence of expected columns
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df_raw.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # Inspect dataset dimensions and target distribution
    n_rows, n_cols = df_raw.shape
    missing_values_count = int(df_raw.isnull().sum().sum())
    duplicate_rows_count = int(df_raw.duplicated().sum())

    target_counts = df_raw[target_col].value_counts().to_dict()
    pos_count = target_counts.get(1, 0)
    neg_count = target_counts.get(0, 0)
    pos_ratio = (pos_count / n_rows) * 100 if n_rows > 0 else 0.0

    metadata = {
        "raw_shape": (n_rows, n_cols),
        "total_observations": n_rows,
        "total_columns": n_cols,
        "missing_values": missing_values_count,
        "duplicate_rows": duplicate_rows_count,
        "target_distribution": {
            "non_failure_count (0)": neg_count,
            "failure_count (1)": pos_count,
            "failure_percentage": round(pos_ratio, 2),
        },
        "dropped_identifier_columns": identifier_cols,
        "dropped_leakage_columns": leakage_cols,
    }

    logger.info(
        f"Raw data loaded: {n_rows} rows, {n_cols} columns. "
        f"Failures: {pos_count} ({pos_ratio:.2f}%), Non-failures: {neg_count}."
    )

    # Strictly drop identifier and leakage columns to prevent data leakage
    cols_to_drop = list(set(identifier_cols + leakage_cols))
    df_clean = df_raw.drop(columns=cols_to_drop, errors="ignore")

    logger.info(
        f"Removed leakage/identifier columns {cols_to_drop}. "
        f"Cleaned dataset shape: {df_clean.shape}."
    )

    return df_clean, metadata


"""
Domain-specific feature engineering for predictive maintenance.
"""

import pandas as pd

from src.utils import setup_logger

logger = setup_logger("feature_engineering")


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate and append domain features.
    
    Engineered Feature:
        temperature_difference = Process temperature [K] - Air temperature [K]
        Captures thermal load and heat accumulation on cutting tool/machine.

    Args:
        df: Input DataFrame containing raw temperature columns.

    Returns:
        DataFrame with engineered feature appended.
    """
    df_out = df.copy()

    air_col = "Air temperature [K]"
    proc_col = "Process temperature [K]"

    if air_col in df_out.columns and proc_col in df_out.columns:
        df_out["temperature_difference"] = df_out[proc_col] - df_out[air_col]
        logger.info("Successfully engineered 'temperature_difference' feature.")
    else:
        logger.warning(
            f"Required columns for temperature_difference not found: {air_col}, {proc_col}"
        )

    return df_out


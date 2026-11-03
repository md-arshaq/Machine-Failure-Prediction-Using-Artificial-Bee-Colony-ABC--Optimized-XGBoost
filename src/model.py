"""
XGBoost model building, training, and serialization.
Compatible with modern XGBoost 2.x/3.x.
"""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.utils import setup_logger

logger = setup_logger("model")


def build_xgb_classifier(
    params: Dict[str, Any],
    random_state: int = 42,
    early_stopping_rounds: Optional[int] = None,
) -> XGBClassifier:
    """
    Instantiate an XGBClassifier with configured parameters.

    Args:
        params: Dictionary of model hyperparameters.
        random_state: Random seed for tree construction.
        early_stopping_rounds: Number of iterations without validation metric improvement.

    Returns:
        Configured XGBClassifier instance.
    """
    model_params = {
        "n_estimators": int(params.get("n_estimators", 300)),
        "max_depth": int(params.get("max_depth", 5)),
        "learning_rate": float(params.get("learning_rate", 0.1)),
        "min_child_weight": float(params.get("min_child_weight", 1.0)),
        "subsample": float(params.get("subsample", 0.8)),
        "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
        "gamma": float(params.get("gamma", 0.0)),
        "scale_pos_weight": float(params.get("scale_pos_weight", 1.0)),
        "reg_lambda": float(params.get("reg_lambda", 1.0)),
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": random_state,
        "n_jobs": -1,
    }

    if early_stopping_rounds is not None and early_stopping_rounds > 0:
        model_params["early_stopping_rounds"] = early_stopping_rounds

    return XGBClassifier(**model_params)


def train_xgb_model(
    model: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    eval_set: Optional[list] = None,
    verbose: bool = False,
) -> XGBClassifier:
    """
    Fit XGBClassifier on training data with optional validation monitoring.

    Args:
        model: XGBClassifier instance.
        X_train: Predictor DataFrame.
        y_train: Target Series.
        eval_set: Optional list of (X, y) evaluation sets for early stopping.
        verbose: Verbosity flag.

    Returns:
        Fitted model.
    """
    if eval_set:
        model.fit(X_train, y_train, eval_set=eval_set, verbose=verbose)
    else:
        model.fit(X_train, y_train, verbose=verbose)
    return model


def save_xgb_model(model: XGBClassifier, filepath: str) -> None:
    """
    Save XGBoost model in standardized JSON format.

    Args:
        model: Fitted XGBClassifier.
        filepath: Output JSON path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save_model(filepath)
    logger.info(f"Model saved to {filepath}")


def load_xgb_model(filepath: str) -> XGBClassifier:
    """
    Load XGBoost model from JSON file.

    Args:
        filepath: Input JSON path.

    Returns:
        Loaded XGBClassifier.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")
    model = XGBClassifier()
    model.load_model(filepath)
    return model


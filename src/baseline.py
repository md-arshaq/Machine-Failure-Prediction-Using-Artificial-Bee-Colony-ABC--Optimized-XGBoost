"""
Baseline XGBoost training and evaluation without optimization.
Uses all 7 logical feature groups and default hyperparameters.
"""

import os
import time
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.metrics import compute_classification_metrics
from src.model import build_xgb_classifier, save_xgb_model, train_xgb_model
from src.preprocessing import Preprocessor
from src.utils import save_json, save_markdown_table, setup_logger

logger = setup_logger("baseline")


def run_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Dict[str, Any],
) -> Tuple[XGBClassifier, Dict[str, Any], Dict[str, Any]]:
    """
    Train and evaluate baseline XGBoost model across validation and test sets.

    Args:
        X_train: Training predictors (all features).
        y_train: Training labels.
        X_val: Validation predictors.
        y_val: Validation labels.
        X_test: Test predictors.
        y_test: Test labels.
        config: Parsed configuration dictionary.

    Returns:
        Tuple of (fitted baseline model, val_metrics dict, test_metrics dict).
    """
    xgb_cfg = config["xgboost"]["baseline"].copy()
    early_stopping_rounds = config["xgboost"].get("early_stopping_rounds", 30)
    models_dir = config["paths"]["models_dir"]
    tables_dir = config["paths"]["tables_dir"]
    preds_dir = config["paths"]["predictions_dir"]
    logs_dir = config["paths"]["logs_dir"]

    # Calculate scale_pos_weight from training set
    scale_pos_weight = Preprocessor.calculate_scale_pos_weight(y_train)
    xgb_cfg["scale_pos_weight"] = scale_pos_weight

    logger.info(
        f"Training Baseline XGBoost with {len(X_train.columns)} features and "
        f"scale_pos_weight={scale_pos_weight:.2f}..."
    )

    model = build_xgb_classifier(
        params=xgb_cfg,
        random_state=xgb_cfg.get("random_state", 42),
        early_stopping_rounds=early_stopping_rounds,
    )

    start_time = time.perf_counter()
    train_xgb_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    training_duration = time.perf_counter() - start_time
    logger.info(f"Baseline training completed in {training_duration:.3f} seconds.")

    # Predictions and metrics on validation set
    val_probs = model.predict_proba(X_val)[:, 1]
    val_metrics = compute_classification_metrics(y_val, val_probs)
    val_metrics["training_time_s"] = training_duration

    # Predictions and metrics on isolated test set
    test_probs = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_classification_metrics(y_test, test_probs)
    test_metrics["training_time_s"] = training_duration

    logger.info(
        f"Baseline Results -> Val AP: {val_metrics['average_precision']:.4f} | "
        f"Val F1: {val_metrics['f1']:.4f} | "
        f"Test AP: {test_metrics['average_precision']:.4f} | "
        f"Test F1: {test_metrics['f1']:.4f}"
    )

    # Save baseline model
    model_path = os.path.join(models_dir, "baseline_xgb.json")
    save_xgb_model(model, model_path)

    # Save test predictions
    os.makedirs(preds_dir, exist_ok=True)
    preds_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_prob": test_probs,
        "y_pred": (test_probs >= 0.5).astype(int),
    })
    preds_path = os.path.join(preds_dir, "baseline_test_predictions.csv")
    preds_df.to_csv(preds_path, index=False)
    logger.info(f"Baseline test predictions saved to {preds_path}")

    # Generate Table 3: Baseline Results
    os.makedirs(tables_dir, exist_ok=True)
    table3_rows = [
        {"Dataset Partition": "Validation Set", "Accuracy": val_metrics["accuracy"], "Precision": val_metrics["precision"], "Recall": val_metrics["recall"], "F1-Score": val_metrics["f1"], "Macro F1": val_metrics["macro_f1"], "ROC-AUC": val_metrics["roc_auc"], "Average Precision (PR-AUC)": val_metrics["average_precision"], "Training Time (s)": round(training_duration, 3)},
        {"Dataset Partition": "Test Set", "Accuracy": test_metrics["accuracy"], "Precision": test_metrics["precision"], "Recall": test_metrics["recall"], "F1-Score": test_metrics["f1"], "Macro F1": test_metrics["macro_f1"], "ROC-AUC": test_metrics["roc_auc"], "Average Precision (PR-AUC)": test_metrics["average_precision"], "Training Time (s)": round(training_duration, 3)},
    ]
    df_tab3 = pd.DataFrame(table3_rows)
    df_tab3.to_csv(os.path.join(tables_dir, "baseline_results.csv"), index=False)
    save_markdown_table(df_tab3, os.path.join(tables_dir, "baseline_results.md"), "Table 3: Baseline XGBoost Classification Performance")

    # Save baseline summary json
    save_json({
        "hyperparameters": xgb_cfg,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "training_duration_s": training_duration,
    }, os.path.join(logs_dir, "baseline_summary.json"))

    return model, val_metrics, test_metrics


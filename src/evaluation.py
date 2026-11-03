"""
Final multi-seed evaluation, model retraining on combined development data,
and generation of Tables 4-8.
"""

import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.metrics import compute_classification_metrics
from src.model import build_xgb_classifier, save_xgb_model, train_xgb_model
from src.preprocessing import Preprocessor
from src.utils import save_json, save_markdown_table, setup_logger

logger = setup_logger("evaluation")


def evaluate_multiseed(
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    baseline_params: Dict[str, Any],
    best_params: Dict[str, Any],
    best_feature_mask: List[bool],
    seeds: List[int],
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Retrain Baseline and ABC-Optimized models on combined dev data across multiple random seeds,
    and evaluate on the isolated test set.

    Args:
        dev_df: Combined Train + Validation DataFrame.
        test_df: Isolated Test DataFrame.
        baseline_params: Default baseline hyperparameter dict.
        best_params: ABC-selected hyperparameter dict.
        best_feature_mask: 7-element boolean mask for selected features.
        seeds: List of integer seeds (e.g., [0, 1, 2, 3, 4]).
        config: Parsed configuration dictionary.

    Returns:
        Tuple of:
          - DataFrame of all per-seed runs.
          - Aggregated comparison DataFrame (mean ± std).
          - Seed 0 predictions dictionary for plotting.
    """
    target_col = config["dataset"]["target_column"]
    models_dir = config["paths"]["models_dir"]
    preds_dir = config["paths"]["predictions_dir"]
    tables_dir = config["paths"]["tables_dir"]

    # Re-fit preprocessing on the combined development set
    prep_dev = Preprocessor(target_col=target_col, random_seed=42)
    X_dev_full, y_dev = prep_dev.fit_transform(dev_df, categorical_col="Type")
    X_test_full, y_test = prep_dev.transform(test_df, categorical_col="Type")

    # Filter columns for ABC model
    selected_cols = prep_dev.get_selected_columns(best_feature_mask)
    X_dev_abc = X_dev_full[selected_cols]
    X_test_abc = X_test_full[selected_cols]

    # Baseline uses all columns
    scale_pos_weight_dev = Preprocessor.calculate_scale_pos_weight(y_dev)
    baseline_params_run = baseline_params.copy()
    baseline_params_run["scale_pos_weight"] = scale_pos_weight_dev

    results_records: List[Dict[str, Any]] = []
    seed0_preds: Dict[str, Any] = {}

    last_abc_model = None

    for seed in seeds:
        logger.info(f"--- Running Final Evaluation for Seed {seed} ---")

        # 1. Train Baseline
        t0 = time.perf_counter()
        base_model = build_xgb_classifier(baseline_params_run, random_state=seed)
        train_xgb_model(base_model, X_dev_full, y_dev)
        base_time = time.perf_counter() - t0

        base_probs = base_model.predict_proba(X_test_full)[:, 1]
        base_metrics = compute_classification_metrics(y_test, base_probs)
        base_metrics.update({"model": "Baseline XGBoost", "seed": seed, "train_time_s": base_time})
        results_records.append(base_metrics)

        # 2. Train ABC-Optimized
        t0 = time.perf_counter()
        abc_model = build_xgb_classifier(best_params, random_state=seed)
        train_xgb_model(abc_model, X_dev_abc, y_dev)
        abc_time = time.perf_counter() - t0
        last_abc_model = abc_model

        abc_probs = abc_model.predict_proba(X_test_abc)[:, 1]
        abc_metrics = compute_classification_metrics(y_test, abc_probs)
        abc_metrics.update({"model": "ABC-Optimized XGBoost", "seed": seed, "train_time_s": abc_time})
        results_records.append(abc_metrics)

        logger.info(
            f"Seed {seed} | Baseline Test AP: {base_metrics['average_precision']:.4f} (F1: {base_metrics['f1']:.4f}) | "
            f"ABC-XGB Test AP: {abc_metrics['average_precision']:.4f} (F1: {abc_metrics['f1']:.4f})"
        )

        if seed == 0:
            seed0_preds = {
                "y_test": y_test.values,
                "base_probs": base_probs,
                "abc_probs": abc_probs,
                "base_model": base_model,
                "abc_model": abc_model,
                "selected_cols": selected_cols,
            }
            # Save predictions
            os.makedirs(preds_dir, exist_ok=True)
            pd.DataFrame({
                "y_true": y_test.values,
                "y_prob": abc_probs,
                "y_pred": (abc_probs >= 0.5).astype(int),
            }).to_csv(os.path.join(preds_dir, "optimized_test_predictions.csv"), index=False)

    # Save final ABC model (seed 0)
    if last_abc_model is not None:
        save_xgb_model(last_abc_model, os.path.join(models_dir, "abc_optimized_xgb.json"))

    df_results = pd.DataFrame(results_records)

    # Table 7: Aggregate mean ± std
    metric_cols = ["accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc", "average_precision"]
    summary_rows = []

    for model_name in ["Baseline XGBoost", "ABC-Optimized XGBoost"]:
        sub = df_results[df_results["model"] == model_name]
        row_dict = {"Model": model_name}
        for m in metric_cols:
            mean_val = sub[m].mean()
            std_val = sub[m].std()
            row_dict[m] = f"{mean_val:.4f} ± {std_val:.4f}"
            row_dict[f"{m}_mean"] = mean_val
            row_dict[f"{m}_std"] = std_val
        summary_rows.append(row_dict)

    df_master = pd.DataFrame(summary_rows)

    # Save Table 7
    os.makedirs(tables_dir, exist_ok=True)
    df_master.to_csv(os.path.join(tables_dir, "master_comparison.csv"), index=False)
    save_markdown_table(
        df_master[["Model", "accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc", "average_precision"]],
        os.path.join(tables_dir, "master_comparison.md"),
        f"Table 7: Master Comparison Across {len(seeds)} Seeds (Mean ± Std)",
    )

    return df_results, df_master, seed0_preds


def generate_tables(
    config: Dict[str, Any],
    best_params: Dict[str, Any],
    best_feature_mask: List[bool],
    best_fitness: float,
    timing_stats: Dict[str, float],
    eval_count: int,
    cache_hits: int,
) -> None:
    """
    Generate Tables 4, 5, 6, and 8.
    """
    tables_dir = config["paths"]["tables_dir"]
    os.makedirs(tables_dir, exist_ok=True)

    # Table 4: ABC Parameters
    abc_cfg = config["abc"]
    df_tab4 = pd.DataFrame([
        {"Parameter": "Number of Food Sources (SN)", "Configured Value": abc_cfg["n_food_sources"], "Description": "Population of candidate solutions"},
        {"Parameter": "Employed Bees", "Configured Value": abc_cfg["n_employed"], "Description": "Assigned 1:1 to food sources for neighborhood exploration"},
        {"Parameter": "Onlooker Bees", "Configured Value": abc_cfg["n_onlooker"], "Description": "Select food sources probabilistically via roulette wheel"},
        {"Parameter": "Abandonment Limit (limit)", "Configured Value": abc_cfg["abandonment_limit"], "Description": "Max consecutive unimproved trials before scout reset"},
        {"Parameter": "Maximum Cycles", "Configured Value": abc_cfg["max_cycles"], "Description": "Total iterations of ABC search loop"},
        {"Parameter": "Maximum Fitness Evaluations", "Configured Value": abc_cfg["max_evals"], "Description": "Hard computational budget limit"},
        {"Parameter": "Random Seed", "Configured Value": abc_cfg["random_seed"], "Description": "Ensures reproducible stochastic search operations"},
    ])
    df_tab4.to_csv(os.path.join(tables_dir, "abc_parameters.csv"), index=False)
    save_markdown_table(df_tab4, os.path.join(tables_dir, "abc_parameters.md"), "Table 4: Artificial Bee Colony Optimization Configuration")

    # Table 5: Best ABC Configuration
    feature_names = [
        "Type (L, M, H)",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "temperature_difference",
    ]
    selected_features_str = ", ".join([f for f, s in zip(feature_names, best_feature_mask) if s])

    df_tab5 = pd.DataFrame([
        {"Hyperparameter / Decision": "Selected Feature Groups", "Optimal Value": selected_features_str},
        {"Hyperparameter / Decision": "n_estimators", "Optimal Value": best_params["n_estimators"]},
        {"Hyperparameter / Decision": "max_depth", "Optimal Value": best_params["max_depth"]},
        {"Hyperparameter / Decision": "learning_rate", "Optimal Value": best_params["learning_rate"]},
        {"Hyperparameter / Decision": "min_child_weight", "Optimal Value": best_params["min_child_weight"]},
        {"Hyperparameter / Decision": "subsample", "Optimal Value": best_params["subsample"]},
        {"Hyperparameter / Decision": "colsample_bytree", "Optimal Value": best_params["colsample_bytree"]},
        {"Hyperparameter / Decision": "gamma", "Optimal Value": best_params["gamma"]},
        {"Hyperparameter / Decision": "scale_pos_weight", "Optimal Value": best_params["scale_pos_weight"]},
        {"Hyperparameter / Decision": "Best Validation AP", "Optimal Value": round(best_fitness, 6)},
    ])
    df_tab5.to_csv(os.path.join(tables_dir, "optimized_results.csv"), index=False)
    save_markdown_table(df_tab5, os.path.join(tables_dir, "optimized_results.md"), "Table 5: ABC Best Discovered Hyperparameters & Feature Mask")

    # Table 8: Runtime Comparison
    total_time = sum(timing_stats.values())
    df_tab8 = pd.DataFrame([
        {"Stage": "Data Loading & Preprocessing", "Time (seconds)": round(timing_stats.get("preprocessing", 0.0), 3), "Notes": "Leakage removal, feature engineering, stratified 70/15/15 split"},
        {"Stage": "EDA & Initial Visualizations", "Time (seconds)": round(timing_stats.get("eda", 0.0), 3), "Notes": "Figures 1-4, Tables 1-2"},
        {"Stage": "Baseline Model Training", "Time (seconds)": round(timing_stats.get("baseline", 0.0), 3), "Notes": "XGBoost baseline on Train/Val/Test"},
        {"Stage": "ABC Optimization Search", "Time (seconds)": round(timing_stats.get("search", 0.0), 3), "Notes": f"{eval_count} actual model evals, {cache_hits} cache hits"},
        {"Stage": "Final Multi-Seed Training (5 seeds)", "Time (seconds)": round(timing_stats.get("final_evaluation", 0.0), 3), "Notes": "Combined dev set retraining (Baseline vs ABC)"},
        {"Stage": "Visualization & Reporting", "Time (seconds)": round(timing_stats.get("visualization", 0.0), 3), "Notes": "Figures 5-15, markdown report"},
        {"Stage": "Total Pipeline Execution", "Time (seconds)": round(total_time, 3), "Notes": f"Overall runtime ({total_time/60:.2f} minutes)"},
    ])
    df_tab8.to_csv(os.path.join(tables_dir, "runtime_comparison.csv"), index=False)
    save_markdown_table(df_tab8, os.path.join(tables_dir, "runtime_comparison.md"), "Table 8: Computational Runtime Breakdown")

    logger.info("Tables 4, 5, and 8 generated successfully.")


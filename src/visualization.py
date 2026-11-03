"""
Comprehensive visualization module.
Generates Figures 5 through 15 at publication quality (150-200 DPI).
"""

import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from src.utils import setup_logger

logger = setup_logger("visualization")


def generate_experiment_plots(
    config: Dict[str, Any],
    eval_log_path: str,
    convergence_log_path: str,
    df_results: pd.DataFrame,
    seed0_preds: Dict[str, Any],
) -> None:
    """
    Generate Figures 5 through 15 using actual experimental results.

    Args:
        config: Parsed configuration dictionary.
        eval_log_path: Path to abc_evaluations.csv.
        convergence_log_path: Path to abc_convergence.csv.
        df_results: DataFrame of 5-seed evaluation results.
        seed0_preds: Dict containing seed 0 predictions, test labels, models, and selected cols.
    """
    fig_dir = config["paths"]["figures_dir"]
    dpi = config["visualization"]["dpi"]
    os.makedirs(fig_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    df_evals = pd.read_csv(eval_log_path) if os.path.exists(eval_log_path) else pd.DataFrame()
    df_conv = pd.read_csv(convergence_log_path) if os.path.exists(convergence_log_path) else pd.DataFrame()

    # -------------------------------------------------------------
    # Figure 5: ABC Convergence Curve
    # -------------------------------------------------------------
    if not df_conv.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            df_conv["cycle"],
            df_conv["best_so_far_ap"],
            marker="o",
            color="#2b5c8f",
            linewidth=2.5,
            label="Best-So-Far Val AP",
        )
        ax.plot(
            df_conv["cycle"],
            df_conv["population_mean_ap"],
            marker="s",
            color="#d95f02",
            linestyle="--",
            linewidth=2.0,
            label="Population Mean Val AP",
        )
        ax.set_title("Artificial Bee Colony Convergence Trajectory", fontsize=13, pad=12)
        ax.set_xlabel("Optimization Cycle")
        ax.set_ylabel("Validation Average Precision (PR-AUC)")
        ax.legend(frameon=True)
        plt.tight_layout()
        fig5_path = os.path.join(fig_dir, "fig05_abc_convergence.png")
        fig.savefig(fig5_path, dpi=dpi)
        plt.close(fig)
        logger.info(f"Saved {fig5_path}")

    # -------------------------------------------------------------
    # Figure 6: Best Fitness Against Evaluation Count
    # -------------------------------------------------------------
    if not df_evals.empty:
        actual_evals_df = df_evals[df_evals["cached"] == False].copy().reset_index(drop=True)
        actual_evals_df["eval_num"] = np.arange(1, len(actual_evals_df) + 1)
        actual_evals_df["running_best_ap"] = actual_evals_df["validation_ap"].cummax()

        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.scatter(
            actual_evals_df["eval_num"],
            actual_evals_df["validation_ap"],
            color="#999999",
            alpha=0.5,
            s=25,
            label="Candidate Val AP",
        )
        ax.step(
            actual_evals_df["eval_num"],
            actual_evals_df["running_best_ap"],
            where="post",
            color="#0066cc",
            linewidth=2.5,
            label="Best-So-Far Val AP",
        )
        ax.set_title("ABC Validation AP vs. Actual Fitness Evaluation Count", fontsize=13, pad=12)
        ax.set_xlabel("Actual Model Evaluations")
        ax.set_ylabel("Validation Average Precision (PR-AUC)")
        ax.legend(frameon=True)
        plt.tight_layout()
        fig6_path = os.path.join(fig_dir, "fig06_best_fitness_vs_evals.png")
        fig.savefig(fig6_path, dpi=dpi)
        plt.close(fig)
        logger.info(f"Saved {fig6_path}")

    # -------------------------------------------------------------
    # Figure 7: ABC Exploration and Exploitation
    # -------------------------------------------------------------
    if not df_conv.empty:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        width = 0.35
        x = np.arange(len(df_conv["cycle"]))
        ax.bar(
            x - width / 2,
            df_conv["improvements"],
            width,
            label="Exploitation (Successful Improvements)",
            color="#2b5c8f",
            edgecolor="black",
        )
        ax.bar(
            x + width / 2,
            df_conv["scout_resets"],
            width,
            label="Exploration (Scout Replacements)",
            color="#d95f02",
            edgecolor="black",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(df_conv["cycle"])
        ax.set_title("ABC Dynamics: Exploration (Scouts) vs. Exploitation (Improvements)", fontsize=13, pad=12)
        ax.set_xlabel("Optimization Cycle")
        ax.set_ylabel("Count per Cycle")
        ax.legend(frameon=True)
        plt.tight_layout()
        fig7_path = os.path.join(fig_dir, "fig07_exploration_exploitation.png")
        fig.savefig(fig7_path, dpi=dpi)
        plt.close(fig)
        logger.info(f"Saved {fig7_path}")

    # -------------------------------------------------------------
    # Figure 8: Baseline vs ABC-XGBoost Metric Comparison
    # -------------------------------------------------------------
    if not df_results.empty:
        metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Average Precision"]

        base_sub = df_results[df_results["model"] == "Baseline XGBoost"]
        abc_sub = df_results[df_results["model"] == "ABC-Optimized XGBoost"]

        base_means = [base_sub[m].mean() for m in metrics_to_plot]
        base_stds = [base_sub[m].std() for m in metrics_to_plot]
        abc_means = [abc_sub[m].mean() for m in metrics_to_plot]
        abc_stds = [abc_sub[m].std() for m in metrics_to_plot]

        x = np.arange(len(metrics_to_plot))
        width = 0.35

        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.bar(
            x - width / 2,
            base_means,
            width,
            yerr=base_stds,
            capsize=4,
            label="Baseline XGBoost",
            color="#7293cb",
            edgecolor="black",
        )
        ax.bar(
            x + width / 2,
            abc_means,
            width,
            yerr=abc_stds,
            capsize=4,
            label="ABC-Optimized XGBoost",
            color="#e1974c",
            edgecolor="black",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.set_ylabel("Test Metric Value")
        ax.set_ylim(0, 1.15)
        ax.set_title("Test Performance Comparison Across 5 Seeds (Mean ± Std)", fontsize=13, pad=12)
        ax.legend(frameon=True, loc="upper right")
        plt.tight_layout()
        fig8_path = os.path.join(fig_dir, "fig08_metric_comparison.png")
        fig.savefig(fig8_path, dpi=dpi)
        plt.close(fig)
        logger.info(f"Saved {fig8_path}")

    # -------------------------------------------------------------
    # Figure 9 & 10: Confusion Matrices (Seed 0)
    # -------------------------------------------------------------
    if "y_test" in seed0_preds:
        y_test = seed0_preds["y_test"]
        base_probs = seed0_preds["base_probs"]
        abc_probs = seed0_preds["abc_probs"]

        base_pred = (base_probs >= 0.5).astype(int)
        abc_pred = (abc_probs >= 0.5).astype(int)

        cm_base = confusion_matrix(y_test, base_pred, labels=[0, 1])
        cm_abc = confusion_matrix(y_test, abc_pred, labels=[0, 1])

        # Figure 9: Baseline CM
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(
            cm_base,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["No Failure (0)", "Failure (1)"],
            yticklabels=["No Failure (0)", "Failure (1)"],
            ax=ax,
        )
        ax.set_title("Baseline XGBoost Confusion Matrix (Test Set)", fontsize=12, pad=10)
        ax.set_ylabel("Actual Ground Truth")
        ax.set_xlabel("Predicted Label")
        plt.tight_layout()
        fig9_path = os.path.join(fig_dir, "fig09_baseline_confusion_matrix.png")
        fig.savefig(fig9_path, dpi=dpi)
        plt.close(fig)
        logger.info(f"Saved {fig9_path}")

        # Figure 10: Optimized CM
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(
            cm_abc,
            annot=True,
            fmt="d",
            cmap="Oranges",
            cbar=False,
            xticklabels=["No Failure (0)", "Failure (1)"],
            yticklabels=["No Failure (0)", "Failure (1)"],
            ax=ax,
        )
        ax.set_title("ABC-Optimized XGBoost Confusion Matrix (Test Set)", fontsize=12, pad=10)
        ax.set_ylabel("Actual Ground Truth")
        ax.set_xlabel("Predicted Label")
        plt.tight_layout()
        fig10_path = os.path.join(fig_dir, "fig10_optimized_confusion_matrix.png")
        fig.savefig(fig10_path, dpi=dpi)
        plt.close(fig)
        logger.info(f"Saved {fig10_path}")

        # -------------------------------------------------------------
        # Figure 11: ROC Curves
        # -------------------------------------------------------------
        fpr_base, tpr_base, _ = roc_curve(y_test, base_probs)
        fpr_abc, tpr_abc, _ = roc_curve(y_test, abc_probs)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr_base, tpr_base, label="Baseline XGBoost", color="#2b5c8f", linewidth=2.0)
        ax.plot(fpr_abc, tpr_abc, label="ABC-Optimized XGBoost", color="#d95f02", linewidth=2.5)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Chance")
        ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, pad=12)
        ax.set_xlabel("False Positive Rate (FPR)")
        ax.set_ylabel("True Positive Rate (TPR)")
        ax.legend(frameon=True, loc="lower right")
        plt.tight_layout()
        fig11_path = os.path.join(fig_dir, "fig11_roc_curves.png")
        fig.savefig(fig11_path, dpi=dpi)
        plt.close(fig)
        logger.info(f"Saved {fig11_path}")

        # -------------------------------------------------------------
        # Figure 12: Precision-Recall Curves
        # -------------------------------------------------------------
        prec_base, rec_base, _ = precision_recall_curve(y_test, base_probs)
        prec_abc, rec_abc, _ = precision_recall_curve(y_test, abc_probs)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(rec_base, prec_base, label="Baseline XGBoost", color="#2b5c8f", linewidth=2.0)
        ax.plot(rec_abc, prec_abc, label="ABC-Optimized XGBoost", color="#d95f02", linewidth=2.5)
        no_skill = sum(y_test) / len(y_test)
        ax.plot([0, 1], [no_skill, no_skill], "k--", alpha=0.5, label=f"No Skill Baseline ({no_skill:.3f})")
        ax.set_title("Precision-Recall (PR) Curves (Primary Imbalanced Metric)", fontsize=13, pad=12)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(frameon=True, loc="lower left")
        plt.tight_layout()
        fig12_path = os.path.join(fig_dir, "fig12_precision_recall_curves.png")
        fig.savefig(fig12_path, dpi=dpi)
        plt.close(fig)
        logger.info(f"Saved {fig12_path}")

        # -------------------------------------------------------------
        # Figure 13: Feature Importance of ABC-Selected Model
        # -------------------------------------------------------------
        abc_model = seed0_preds.get("abc_model")
        selected_cols = seed0_preds.get("selected_cols", [])
        if abc_model and selected_cols:
            importances = abc_model.feature_importances_
            feat_imp = pd.Series(importances, index=selected_cols).sort_values(ascending=True)

            fig, ax = plt.subplots(figsize=(8, max(4.5, len(selected_cols) * 0.45)))
            feat_imp.plot(kind="barh", color="#2ca02c", edgecolor="black", ax=ax)
            ax.set_title("XGBoost Gain-Based Feature Importance (ABC Selected)", fontsize=12, pad=10)
            ax.set_xlabel("Relative Importance (Gain)")
            plt.tight_layout()
            fig13_path = os.path.join(fig_dir, "fig13_feature_importance.png")
            fig.savefig(fig13_path, dpi=dpi)
            plt.close(fig)
            logger.info(f"Saved {fig13_path}")

    # -------------------------------------------------------------
    # Figure 14: Feature Selection Frequency Across Search Evaluations
    # -------------------------------------------------------------
    if not df_evals.empty and "selected_features" in df_evals.columns:
        all_features = [
            "Type",
            "Air_temp",
            "Process_temp",
            "Rotational_speed",
            "Torque",
            "Tool_wear",
            "temp_diff",
        ]
        counts = {f: 0 for f in all_features}
        for sel_str in df_evals["selected_features"].dropna():
            parts = str(sel_str).split("|")
            for p in parts:
                if p in counts:
                    counts[p] += 1

        total_runs = len(df_evals)
        pct_counts = {k: (v / total_runs) * 100 for k, v in counts.items()}
        s_pct = pd.Series(pct_counts).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        s_pct.plot(kind="barh", color="#17becf", edgecolor="black", ax=ax)
        ax.set_title("Feature Group Selection Frequency Across All ABC Evaluations", fontsize=12, pad=10)
        ax.set_xlabel("Selection Frequency (% of total candidate evaluations)")
        ax.set_xlim(0, 100)
        plt.tight_layout()
        fig14_path = os.path.join(fig_dir, "fig14_feature_selection_frequency.png")
        fig.savefig(fig14_path, dpi=dpi)
        plt.close(fig)
        logger.info(f"Saved {fig14_path}")

    # -------------------------------------------------------------
    # Figure 15: Optimization Search Analysis
    # -------------------------------------------------------------
    if not df_evals.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        sns.histplot(df_evals["validation_ap"], bins=20, kde=True, color="#9467bd", ax=ax1)
        ax1.set_title("Distribution of Candidate Validation AP Scores", fontsize=12)
        ax1.set_xlabel("Validation AP (PR-AUC)")
        ax1.set_ylabel("Candidate Count")

        # Scatter of learning_rate vs validation_ap
        if "learning_rate" in df_evals.columns and "max_depth" in df_evals.columns:
            sc = ax2.scatter(
                df_evals["learning_rate"],
                df_evals["validation_ap"],
                c=df_evals["max_depth"],
                cmap="viridis",
                s=35,
                alpha=0.8,
            )
            cbar = plt.colorbar(sc, ax=ax2)
            cbar.set_label("max_depth")
            ax2.set_xscale("log")
            ax2.set_title("Validation AP vs. Learning Rate (Color: max_depth)", fontsize=12)
            ax2.set_xlabel("Learning Rate (Log Scale)")
            ax2.set_ylabel("Validation AP")

        plt.tight_layout()
        fig15_path = os.path.join(fig_dir, "fig15_search_analysis.png")
        fig.savefig(fig15_path, dpi=dpi)
        plt.close(fig)
        logger.info(f"Saved {fig15_path}")

    logger.info("All Figures (5 through 15) successfully generated.")


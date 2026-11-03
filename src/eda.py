"""
Exploratory Data Analysis (EDA) module.
Generates publication-quality figures (Figures 1-4) and summary tables (Tables 1-2).
"""

import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import save_markdown_table, setup_logger

logger = setup_logger("eda")


def run_eda(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    df_fe: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
) -> None:
    """
    Execute full exploratory data analysis, saving Figures 1-4 and Tables 1-2.

    Args:
        df_raw: Raw AI4I 2020 DataFrame.
        df_clean: Cleaned DataFrame (leakage columns removed).
        df_fe: Feature engineered DataFrame.
        train_df: Training set DataFrame.
        val_df: Validation set DataFrame.
        test_df: Test set DataFrame.
        config: Parsed configuration dictionary.
    """
    fig_dir = config["paths"]["figures_dir"]
    tab_dir = config["paths"]["tables_dir"]
    dpi = config["visualization"]["dpi"]
    target_col = config["dataset"]["target_column"]

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=1.1)

    logger.info("Generating Figure 1: Target Class Distribution...")
    # Figure 1: Target Class Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df_raw[target_col].value_counts().sort_index()
    labels = ["No Failure (0)", "Failure (1)"]
    colors = ["#2b5c8f", "#d95f02"]
    bars = ax.bar(labels, counts.values, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df_raw)) * 100
        ax.annotate(
            f"{h:,}\n({pct:.2f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_title("AI4I 2020 Machine Failure Class Distribution (Target Imbalance)", fontsize=13, pad=15)
    ax.set_ylabel("Number of Observations")
    ax.set_ylim(0, len(df_raw) * 1.12)
    plt.tight_layout()
    fig1_path = os.path.join(fig_dir, "fig01_target_distribution.png")
    fig.savefig(fig1_path, dpi=dpi)
    plt.close(fig)
    logger.info(f"Saved {fig1_path}")

    logger.info("Generating Figure 2: Feature Correlation Heatmap...")
    # Figure 2: Feature Correlation Heatmap
    num_cols = config["dataset"]["numerical_features"] + config["dataset"]["engineered_features"]
    corr_matrix = df_fe[num_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6.5))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.8,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap of Numerical & Engineered Features", fontsize=13, pad=15)
    plt.tight_layout()
    fig2_path = os.path.join(fig_dir, "fig02_correlation_heatmap.png")
    fig.savefig(fig2_path, dpi=dpi)
    plt.close(fig)
    logger.info(f"Saved {fig2_path}")

    logger.info("Generating Figure 3: Feature Distributions by Failure State...")
    # Figure 3: Numerical Feature Distributions
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for idx, col in enumerate(num_cols):
        ax = axes[idx]
        sns.kdeplot(
            data=df_fe,
            x=col,
            hue=target_col,
            common_norm=False,
            palette=["#2b5c8f", "#d95f02"],
            fill=True,
            alpha=0.3,
            ax=ax,
        )
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_ylabel("Density")
        if ax.get_legend():
            ax.get_legend().remove()
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#2b5c8f", alpha=0.6, label="No Failure (0)"),
        plt.Rectangle((0, 0), 1, 1, color="#d95f02", alpha=0.6, label="Failure (1)"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.95, 0.98), frameon=True)
    fig.suptitle("Distributions of Numerical Features Stratified by Failure State", fontsize=14, y=1.02)
    plt.tight_layout()
    fig3_path = os.path.join(fig_dir, "fig03_feature_distributions.png")
    fig.savefig(fig3_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {fig3_path}")

    logger.info("Generating Figure 4: Failure Distribution by Product Type...")
    # Figure 4: Failure by Product Type
    type_stats = (
        df_raw.groupby(["Type", target_col])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "No Failure", 1: "Failure"})
    )
    type_stats["Failure Rate (%)"] = (
        type_stats["Failure"] / (type_stats["No Failure"] + type_stats["Failure"])
    ) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    type_stats[["No Failure", "Failure"]].plot(
        kind="bar",
        stacked=True,
        color=["#2b5c8f", "#d95f02"],
        ax=ax1,
        edgecolor="black",
        linewidth=1.0,
    )
    ax1.set_title("Machine Observations by Product Type (L, M, H)", fontsize=12)
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Product Quality Category")
    ax1.tick_params(axis="x", rotation=0)
    ax1.legend(loc="upper right")

    bars = ax2.bar(
        type_stats.index,
        type_stats["Failure Rate (%)"],
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
        edgecolor="black",
        linewidth=1.0,
        width=0.5,
    )
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(
            f"{h:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax2.set_title("Failure Rate (%) by Product Type", fontsize=12)
    ax2.set_ylabel("Failure Percentage (%)")
    ax2.set_xlabel("Product Quality Category")
    ax2.set_ylim(0, max(type_stats["Failure Rate (%)"]) * 1.25)

    plt.tight_layout()
    fig4_path = os.path.join(fig_dir, "fig04_failure_by_type.png")
    fig.savefig(fig4_path, dpi=dpi)
    plt.close(fig)
    logger.info(f"Saved {fig4_path}")

    # Table 1: Dataset Summary
    logger.info("Generating Table 1: Dataset Summary...")
    table1_data = [
        {"Feature": "UDI", "Original Type": "Integer", "Usage Decision": "Dropped", "Rationale": "Arbitrary sequential index; zero predictive utility"},
        {"Feature": "Product ID", "Original Type": "Categorical", "Usage Decision": "Dropped", "Rationale": "Product serial code (redundant with Type; high cardinality)"},
        {"Feature": "Type", "Original Type": "Categorical", "Usage Decision": "Retained (OHE)", "Rationale": "Product quality variant (Low, Medium, High) affecting failure risk"},
        {"Feature": "Air temperature [K]", "Original Type": "Numerical", "Usage Decision": "Retained", "Rationale": "Ambient operating temperature recorded by sensors"},
        {"Feature": "Process temperature [K]", "Original Type": "Numerical", "Usage Decision": "Retained", "Rationale": "Direct machinery operational process temperature"},
        {"Feature": "Rotational speed [rpm]", "Original Type": "Numerical", "Usage Decision": "Retained", "Rationale": "Spindle rotational velocity in revolutions per minute"},
        {"Feature": "Torque [Nm]", "Original Type": "Numerical", "Usage Decision": "Retained", "Rationale": "Motor torque delivered to the cutting head"},
        {"Feature": "Tool wear [min]", "Original Type": "Numerical", "Usage Decision": "Retained", "Rationale": "Cumulative cutting tool operating time in minutes"},
        {"Feature": "Machine failure", "Original Type": "Binary", "Usage Decision": "Target Variable", "Rationale": "Primary classification target (0 = Normal, 1 = Failure)"},
        {"Feature": "TWF", "Original Type": "Binary", "Usage Decision": "Dropped (Leakage)", "Rationale": "Tool Wear Failure indicator; directly reveals machine failure"},
        {"Feature": "HDF", "Original Type": "Binary", "Usage Decision": "Dropped (Leakage)", "Rationale": "Heat Dissipation Failure indicator; directly reveals machine failure"},
        {"Feature": "PWF", "Original Type": "Binary", "Usage Decision": "Dropped (Leakage)", "Rationale": "Power Failure indicator; directly reveals machine failure"},
        {"Feature": "OSF", "Original Type": "Binary", "Usage Decision": "Dropped (Leakage)", "Rationale": "Overstrain Failure indicator; directly reveals machine failure"},
        {"Feature": "RNF", "Original Type": "Binary", "Usage Decision": "Dropped (Leakage)", "Rationale": "Random Failure indicator; directly reveals machine failure"},
        {"Feature": "temperature_difference", "Original Type": "Numerical (Engineered)", "Usage Decision": "Candidate Feature", "Rationale": "Process Temp - Air Temp; captures machine thermal dissipation deficit"},
    ]
    df_tab1 = pd.DataFrame(table1_data)
    df_tab1.to_csv(os.path.join(tab_dir, "dataset_summary.csv"), index=False)
    save_markdown_table(df_tab1, os.path.join(tab_dir, "dataset_summary.md"), "Table 1: AI4I 2020 Dataset Summary and Feature Decisions")

    # Table 2: Preprocessing Summary
    logger.info("Generating Table 2: Preprocessing Summary...")
    train_fail = int(train_df[target_col].sum())
    val_fail = int(val_df[target_col].sum())
    test_fail = int(test_df[target_col].sum())
    table2_data = [
        {"Preprocessing Step": "Dataset Filtering", "Specification": "Removed 2 identifier columns (UDI, Product ID) and 5 leakage failure-mode indicators (TWF, HDF, PWF, OSF, RNF)"},
        {"Preprocessing Step": "Feature Engineering", "Specification": "Engineered 'temperature_difference' = Process temperature [K] - Air temperature [K]"},
        {"Preprocessing Step": "Categorical Encoding", "Specification": "OneHotEncoder fitted exclusively on training split; generated Type_H, Type_L, Type_M"},
        {"Preprocessing Step": "Logical Grouping", "Specification": "7 logical groups; Type dummy columns treated as an indivisible single block (Group 0)"},
        {"Preprocessing Step": "Data Partitioning", "Specification": f"Stratified 70/15/15: Train={len(train_df)} ({train_fail} failures, {train_fail/len(train_df)*100:.2f}%), Val={len(val_df)} ({val_fail} failures, {val_fail/len(val_df)*100:.2f}%), Test={len(test_df)} ({test_fail} failures, {test_fail/len(test_df)*100:.2f}%)"},
        {"Preprocessing Step": "Class Imbalance Strategy", "Specification": f"scale_pos_weight = N_negative / N_positive = {(len(train_df)-train_fail)/train_fail:.2f} on training set for baseline; bounded tuning [1, 30] in ABC"},
    ]
    df_tab2 = pd.DataFrame(table2_data)
    df_tab2.to_csv(os.path.join(tab_dir, "preprocessing_summary.csv"), index=False)
    save_markdown_table(df_tab2, os.path.join(tab_dir, "preprocessing_summary.md"), "Table 2: Preprocessing and Data Pipeline Summary")

    logger.info("EDA completed successfully! Figures 1-4 and Tables 1-2 generated.")


# Machine Failure Prediction Using Artificial Bee Colony (ABC)-Optimized XGBoost

**Course:** Bio-Inspired Algorithms (`23DS7BSBIO`)  
**Department:** Computer Science and Engineering (Data Science)  
**Institution:** B.M.S. College of Engineering, Bengaluru  
**Dataset:** UCI AI4I 2020 Predictive Maintenance Dataset (`ai4i2020.csv`)  
**Target:** `Machine failure` (Imbalanced binary classification, ~3.39% positive)  

---

## 1. Project Overview

This project develops an end-to-end predictive maintenance system that forecasts industrial machinery failures from multivariate sensor telemetry. An **XGBoost classification model** serves as the core predictor, and its performance is optimized using a custom **Artificial Bee Colony (ABC)** metaheuristic algorithm implemented **from scratch in NumPy** (no third-party optimization libraries).

The ABC optimizer jointly optimizes:
1. **Feature Selection:** Identifying the most informative subset among 7 logical feature groups (Type, Air Temp, Process Temp, Rotational Speed, Torque, Tool Wear, and engineered Temperature Difference).
2. **Hyperparameter Optimization:** Tuning 8 critical regularization, capacity, and class-weighting hyperparameters of XGBoost.

Optimization is guided by **validation Average Precision (PR-AUC)** to rigorously address extreme class imbalance (~3.39% failure rate). Strict **data leakage prevention** is enforced by removing direct failure indicators (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) and identifiers (`UDI`, `Product ID`).

---

## 2. Directory Structure

```text
BIO-AAT/
├── config.yaml                 # Centralized configuration (dataset, ABC, XGBoost, paths)
├── requirements.txt            # Project dependencies
├── README.md                   # Setup, execution guide, and output mapping
├── PROJECT_CONTEXT.md          # Complete project specification and rubric requirements
├── run_experiments.py          # Unified CLI entry point
├── run_colab.ipynb             # Ready-to-run Google Colab / Jupyter notebook
│
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv        # Raw dataset from UCI Machine Learning Repository
│   └── processed/
│
├── src/
│   ├── __init__.py
│   ├── utils.py                # Logging, seed management, timers, atomic JSON/CSV I/O
│   ├── data_loader.py          # Data validation and leakage stripping
│   ├── feature_engineering.py  # Domain-specific feature engineering (temperature difference)
│   ├── preprocessing.py        # Stratified 70/15/15 splitting, categorical OHE, group mapping
│   ├── eda.py                  # Exploratory data analysis (Figures 1-4, Tables 1-2)
│   ├── metrics.py              # AP / PR-AUC, ROC-AUC, F1, Precision, Recall, Confusion Matrix
│   ├── model.py                # Standardized XGBoost classifier (CPU hist method)
│   ├── baseline.py             # Baseline model training and evaluation (Table 3, Fig 9)
│   ├── search_space.py         # 15D vector [0,1]^15 encoding/decoding and boundary repair
│   ├── fitness.py              # Validation AP evaluation, caching, and evaluation logging
│   ├── abc_optimizer.py        # 100% scratch NumPy ABC (Employed, Onlooker, Scout bees)
│   ├── evaluation.py           # Multi-seed final retraining on combined dev data, Tables 4-8
│   ├── visualization.py        # Publication-quality plotting for Figures 5-15
│   └── experiment.py           # Stage orchestration and Markdown report generation
│
├── results/
│   ├── logs/                   # Evaluation logs (abc_evaluations.csv, abc_convergence.csv)
│   ├── tables/                 # Tables 1-8 in CSV and Markdown formats
│   ├── figures/                # Figures 1-15 (PNG at 200 DPI)
│   ├── models/                 # Saved models (baseline_xgb.json, abc_optimized_xgb.json)
│   ├── predictions/            # Test set predictions (baseline & optimized)
│   └── summary.md              # Auto-generated academic findings report
│
├── checkpoints/                # Persistent recovery states (abc_state.json, fitness_cache.json)
└── tests/
    ├── test_data_pipeline.py   # Unit tests for data loading and leakage prevention
    └── test_abc.py             # Unit tests for ABC operators, decoding, and caching
```

---

## 3. Installation & Setup

### Prerequisites
* Python 3.10+ (tested on Python 3.11)
* Standard CPU (no GPU required)

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 4. Execution Commands

### A. Quick Smoke Test (~1-2 minutes)
Exercises all data pipelines, baseline training, ABC optimization (POP=4, CYCLES=2), multi-seed retraining, and generates all 15 plots and 8 tables:
```bash
python run_experiments.py --stage all --quick
```

### B. Full Experimental Pipeline (~3-5 minutes on CPU)
Runs the full optimization (10 food sources, 10 cycles, up to 240 evaluations) and multi-seed evaluation across 5 random seeds (`[0, 1, 2, 3, 4]`):
```bash
python run_experiments.py --stage all
```

### C. Individual Stage Execution
Run individual stages modularly:
```bash
# 1. Exploratory Data Analysis (Figures 1-4, Tables 1-2)
python run_experiments.py --stage eda

# 2. Baseline XGBoost Model (Table 3, Figure 9)
python run_experiments.py --stage baseline

# 3. Artificial Bee Colony Optimization (Logs, Checkpoints, Best Config)
python run_experiments.py --stage search

# 4. Multi-Seed Retraining (5 seeds on combined Dev set, Table 7)
python run_experiments.py --stage final

# 5. Visualizations & Tables (Figures 5-15, Tables 4, 5, 8, and summary.md)
python run_experiments.py --stage plots
```

### D. Checkpoint Resume
If interrupted, resume the ABC optimization seamlessly from the last completed cycle:
```bash
python run_experiments.py --stage search --resume
```

---

## 5. Verification & Unit Tests

Run the test suite to verify algorithm correctness, boundary constraints, and data leakage isolation:
```bash
python tests/test_data_pipeline.py
python tests/test_abc.py
```

---

## 6. Mapping Outputs to Academic Report Sections

| Report / Rubric Section | Primary Generated Artifacts | Description |
| :--- | :--- | :--- |
| **Problem & Dataset (4 Marks)** | `results/figures/fig01_target_distribution.png`<br>`results/figures/fig02_correlation_heatmap.png`<br>`results/figures/fig03_feature_distributions.png`<br>`results/figures/fig04_failure_by_type.png`<br>`results/tables/dataset_summary.md`<br>`results/tables/preprocessing_summary.md` | EDA, target imbalance visualization, correlation analysis, and complete leakage column audit. |
| **Model & Baseline (6 Marks)** | `results/models/baseline_xgb.json`<br>`results/tables/baseline_results.md`<br>`results/figures/fig09_baseline_confusion_matrix.png` | Baseline XGBoost architecture, unoptimized validation/test metrics, and confusion matrix. |
| **Nature-Inspired ABC (10 Marks)** | `src/abc_optimizer.py`<br>`results/tables/abc_parameters.md`<br>`results/tables/optimized_results.md`<br>`results/logs/abc_convergence.csv`<br>`results/logs/abc_evaluations.csv`<br>`results/figures/fig05_abc_convergence.png`<br>`results/figures/fig06_best_fitness_vs_evals.png`<br>`results/figures/fig07_exploration_exploitation.png` | Pure NumPy ABC implementation, Employed/Onlooker/Scout dynamics, convergence curve, and optimal hyperparameters. |
| **Results & Visualization (6 Marks)** | `results/tables/master_comparison.md`<br>`results/figures/fig08_metric_comparison.png`<br>`results/figures/fig10_optimized_confusion_matrix.png`<br>`results/figures/fig11_roc_curves.png`<br>`results/figures/fig12_precision_recall_curves.png`<br>`results/figures/fig13_feature_importance.png`<br>`results/figures/fig14_feature_selection_frequency.png`<br>`results/figures/fig15_search_analysis.png` | 5-seed performance comparison (mean ± std), ROC & PR curves, confusion matrix comparison, and feature importance. |
| **Report & Code Quality (4 Marks)** | `results/summary.md`<br>`results/tables/runtime_comparison.md`<br>`tests/` | Comprehensive academic summary, computational cost breakdown, modular type-hinted code, and unit tests. |


"""
Main experimental orchestration module.
Coordinates data preprocessing, EDA, baseline evaluation, ABC search, final multi-seed retraining,
visualization generation, and academic summary reporting.
"""

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.abc_optimizer import ArtificialBeeColony
from src.baseline import run_baseline
from src.data_loader import load_raw_data
from src.eda import run_eda
from src.evaluation import evaluate_multiseed, generate_tables
from src.feature_engineering import add_engineered_features
from src.fitness import FitnessEvaluator
from src.preprocessing import Preprocessor
from src.search_space import SearchSpace
from src.utils import load_json, save_json, setup_logger
from src.visualization import generate_experiment_plots

logger = setup_logger("experiment")


class ExperimentRunner:
    """
    Coordinates end-to-end execution across CLI stages.
    """

    def __init__(self, config: Dict[str, Any], quick_mode: bool = False):
        self.config = config
        self.quick_mode = quick_mode
        self.timing_stats: Dict[str, float] = {}

        # Adjust parameters if quick_mode enabled
        if self.quick_mode:
            logger.info("QUICK MODE ACTIVE: using minimal population, cycles, and evaluation budget.")
            q_cfg = self.config["abc"]["quick_mode"]
            self.config["abc"]["n_food_sources"] = q_cfg["n_food_sources"]
            self.config["abc"]["n_employed"] = q_cfg["n_employed"]
            self.config["abc"]["n_onlooker"] = q_cfg["n_onlooker"]
            self.config["abc"]["max_cycles"] = q_cfg["max_cycles"]
            self.config["abc"]["abandonment_limit"] = q_cfg["abandonment_limit"]
            self.config["abc"]["max_evals"] = q_cfg["max_evals"]
            self.config["xgboost"]["baseline"]["n_estimators"] = 50
            self.config["xgboost"]["early_stopping_rounds"] = 10
            self.config["xgboost"]["final_seeds"] = [0, 1]

        # Prepare directory structure
        for path_key in self.config["paths"].values():
            os.makedirs(path_key, exist_ok=True)

        # Cached datasets
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_clean: Optional[pd.DataFrame] = None
        self.df_fe: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.preprocessor: Optional[Preprocessor] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.X_val: Optional[pd.DataFrame] = None
        self.y_val: Optional[pd.Series] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.Series] = None

    def prepare_data(self) -> None:
        """Load, clean, engineer, and partition dataset."""
        if self.X_train is not None:
            return  # Already loaded

        t0 = time.perf_counter()
        ds_cfg = self.config["dataset"]

        df_clean, meta = load_raw_data(
            filepath=ds_cfg["raw_path"],
            identifier_cols=ds_cfg["identifier_columns"],
            leakage_cols=ds_cfg["leakage_columns"],
            target_col=ds_cfg["target_column"],
        )
        self.df_raw = pd.read_csv(ds_cfg["raw_path"])
        self.df_clean = df_clean
        self.df_fe = add_engineered_features(df_clean)

        self.preprocessor = Preprocessor(
            target_col=ds_cfg["target_column"],
            random_seed=ds_cfg["random_seed"],
        )
        train_df, val_df, test_df = self.preprocessor.stratified_split(
            self.df_fe,
            train_ratio=ds_cfg["train_ratio"],
            val_ratio=ds_cfg["val_ratio"],
            test_ratio=ds_cfg["test_ratio"],
        )
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.X_train, self.y_train = self.preprocessor.fit_transform(train_df, categorical_col="Type")
        self.X_val, self.y_val = self.preprocessor.transform(val_df, categorical_col="Type")
        self.X_test, self.y_test = self.preprocessor.transform(test_df, categorical_col="Type")

        self.timing_stats["preprocessing"] = time.perf_counter() - t0

    def run_stage_eda(self) -> None:
        """Stage: Exploratory Data Analysis."""
        self.prepare_data()
        t0 = time.perf_counter()
        logger.info("=== Running Stage: Exploratory Data Analysis (EDA) ===")
        run_eda(
            df_raw=self.df_raw,
            df_clean=self.df_clean,
            df_fe=self.df_fe,
            train_df=self.train_df,
            val_df=self.val_df,
            test_df=self.test_df,
            config=self.config,
        )
        self.timing_stats["eda"] = time.perf_counter() - t0

    def run_stage_baseline(self) -> None:
        """Stage: Baseline XGBoost Evaluation."""
        self.prepare_data()
        t0 = time.perf_counter()
        logger.info("=== Running Stage: Baseline XGBoost Model ===")
        run_baseline(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            X_test=self.X_test,
            y_test=self.y_test,
            config=self.config,
        )
        self.timing_stats["baseline"] = time.perf_counter() - t0

    def run_stage_search(self, resume: bool = False) -> None:
        """Stage: Artificial Bee Colony Optimization."""
        self.prepare_data()
        t0 = time.perf_counter()
        logger.info(f"=== Running Stage: ABC Optimization (Resume={resume}) ===")

        abc_cfg = self.config["abc"]
        xgb_bounds = self.config["xgboost"]["search_bounds"]
        n_est_cap = 50 if self.quick_mode else 600

        search_space = SearchSpace(bounds_config=xgb_bounds, n_estimators_cap=n_est_cap)

        evaluator = FitnessEvaluator(
            X_train_full=self.X_train,
            y_train=self.y_train,
            X_val_full=self.X_val,
            y_val=self.y_val,
            preprocessor=self.preprocessor,
            search_space=search_space,
            max_evals=abc_cfg["max_evals"],
            cache_path=os.path.join(self.config["paths"]["checkpoints_dir"], "fitness_cache.json"),
            eval_log_path=os.path.join(self.config["paths"]["logs_dir"], "abc_evaluations.csv"),
            early_stopping_rounds=self.config["xgboost"]["early_stopping_rounds"],
            random_seed=abc_cfg["random_seed"],
        )

        abc = ArtificialBeeColony(
            evaluator=evaluator,
            n_food_sources=abc_cfg["n_food_sources"],
            n_onlooker=abc_cfg["n_onlooker"],
            max_cycles=abc_cfg["max_cycles"],
            abandonment_limit=abc_cfg["abandonment_limit"],
            random_seed=abc_cfg["random_seed"],
            checkpoint_path=os.path.join(self.config["paths"]["checkpoints_dir"], "abc_state.json"),
            convergence_path=os.path.join(self.config["paths"]["logs_dir"], "abc_convergence.csv"),
        )

        best_vector, best_fitness, cycle_history = abc.optimize(resume=resume)

        # Decode optimal configuration
        best_feature_mask, best_hyperparams = search_space.decode(best_vector)

        # Save optimal results
        models_dir = self.config["paths"]["models_dir"]
        logs_dir = self.config["paths"]["logs_dir"]

        save_json(best_hyperparams, os.path.join(models_dir, "abc_best_config.json"))
        save_json(best_feature_mask, os.path.join(models_dir, "abc_feature_mask.json"))
        save_json({
            "best_validation_ap": best_fitness,
            "best_hyperparameters": best_hyperparams,
            "best_feature_mask": best_feature_mask,
            "actual_evaluations": evaluator.eval_counter,
            "cache_hits": evaluator.cache_hits,
            "total_requests": evaluator.total_requests,
        }, os.path.join(logs_dir, "abc_summary.json"))

        self.timing_stats["search"] = time.perf_counter() - t0
        logger.info(
            f"ABC Optimization Complete! Best Val AP: {best_fitness:.4f} "
            f"across {evaluator.eval_counter} actual evaluations ({evaluator.cache_hits} cache hits)."
        )

    def run_stage_final(self) -> None:
        """Stage: Final Multi-Seed Retraining on Combined Dev Set."""
        self.prepare_data()
        t0 = time.perf_counter()
        logger.info("=== Running Stage: Final Multi-Seed Model Evaluation ===")

        models_dir = self.config["paths"]["models_dir"]
        logs_dir = self.config["paths"]["logs_dir"]

        best_config_path = os.path.join(models_dir, "abc_best_config.json")
        best_mask_path = os.path.join(models_dir, "abc_feature_mask.json")

        if not os.path.exists(best_config_path) or not os.path.exists(best_mask_path):
            raise FileNotFoundError(
                "ABC best configuration not found. Please run '--stage search' first!"
            )

        best_params = load_json(best_config_path)
        best_feature_mask = load_json(best_mask_path)

        # Combine Train and Validation into Development set
        dev_df = pd.concat([self.train_df, self.val_df], ignore_index=True)

        seeds = self.config["xgboost"]["final_seeds"]
        baseline_params = self.config["xgboost"]["baseline"]

        df_results, df_master, seed0_preds = evaluate_multiseed(
            dev_df=dev_df,
            test_df=self.test_df,
            baseline_params=baseline_params,
            best_params=best_params,
            best_feature_mask=best_feature_mask,
            seeds=seeds,
            config=self.config,
        )

        self.timing_stats["final_evaluation"] = time.perf_counter() - t0

        # Cache seed0_preds for plotting stage
        self.df_results = df_results
        self.seed0_preds = seed0_preds

    def run_stage_plots(self) -> None:
        """Stage: Generate Visualizations and Output Tables."""
        self.prepare_data()
        t0 = time.perf_counter()
        logger.info("=== Running Stage: Visualization & Reporting ===")

        models_dir = self.config["paths"]["models_dir"]
        logs_dir = self.config["paths"]["logs_dir"]
        tables_dir = self.config["paths"]["tables_dir"]

        best_config_path = os.path.join(models_dir, "abc_best_config.json")
        best_mask_path = os.path.join(models_dir, "abc_feature_mask.json")
        abc_summary_path = os.path.join(logs_dir, "abc_summary.json")

        if not os.path.exists(best_config_path):
            raise FileNotFoundError("Run 'search' and 'final' stages before generating plots.")

        best_params = load_json(best_config_path)
        best_feature_mask = load_json(best_mask_path)
        abc_summary = load_json(abc_summary_path)

        # Load 5-seed master results if available
        master_csv = os.path.join(tables_dir, "master_comparison.csv")
        df_results = self.df_results if hasattr(self, "df_results") else pd.DataFrame()

        # If seed0_preds not in memory, recompute or load
        if not hasattr(self, "seed0_preds"):
            dev_df = pd.concat([self.train_df, self.val_df], ignore_index=True)
            _, _, seed0_preds = evaluate_multiseed(
                dev_df=dev_df,
                test_df=self.test_df,
                baseline_params=self.config["xgboost"]["baseline"],
                best_params=best_params,
                best_feature_mask=best_feature_mask,
                seeds=[0],
                config=self.config,
            )
        else:
            seed0_preds = self.seed0_preds

        # Generate Figures 5 through 15
        generate_experiment_plots(
            config=self.config,
            eval_log_path=os.path.join(logs_dir, "abc_evaluations.csv"),
            convergence_log_path=os.path.join(logs_dir, "abc_convergence.csv"),
            df_results=df_results,
            seed0_preds=seed0_preds,
        )

        # Generate Tables 4, 5, 8
        generate_tables(
            config=self.config,
            best_params=best_params,
            best_feature_mask=best_feature_mask,
            best_fitness=abc_summary.get("best_validation_ap", 0.0),
            timing_stats=self.timing_stats,
            eval_count=abc_summary.get("actual_evaluations", 0),
            cache_hits=abc_summary.get("cache_hits", 0),
        )

        self.timing_stats["visualization"] = time.perf_counter() - t0

        # Generate results/summary.md
        self.generate_summary_report(best_params, best_feature_mask, abc_summary)

    def generate_summary_report(
        self,
        best_params: Dict[str, Any],
        best_feature_mask: List[bool],
        abc_summary: Dict[str, Any],
    ) -> None:
        """Create final academic Markdown report mapped to course rubric."""
        summary_path = os.path.join(self.config["paths"]["results_dir"], "summary.md")
        master_table_path = os.path.join(self.config["paths"]["tables_dir"], "master_comparison.md")
        master_table_content = ""
        if os.path.exists(master_table_path):
            with open(master_table_path, "r", encoding="utf-8") as f:
                master_table_content = f.read()

        feature_names = [
            "Type (L, M, H)",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "temperature_difference",
        ]
        selected_features = [f for f, s in zip(feature_names, best_feature_mask) if s]

        content = f"""# Experimental Report: ABC-Optimized XGBoost for Machine Failure Prediction

**Course:** Bio-Inspired Algorithms (23DS7BSBIO)  
**Institution:** B.M.S. College of Engineering, Bengaluru  
**Dataset:** AI4I 2020 Predictive Maintenance Dataset (UCI Repository)  
**Target:** `Machine failure` (Imbalanced binary classification, ~3.39% positive)  

---

## 1. Project Overview & Problem Definition
Industrial machinery operates continuously under complex mechanical and thermal stresses. Sudden equipment failure causes catastrophic production downtime, costly reactive maintenance, and safety hazards. Predictive maintenance seeks to forecast impending failures from operating telemetry before catastrophic breakdown.

In this investigation:
1. An **XGBoost classification model** was developed to predict machine failure from sensor telemetry.
2. A **custom Artificial Bee Colony (ABC) optimizer** was developed from scratch in pure NumPy to jointly optimize:
   - **Feature Selection:** Identifying the minimal, most informative subset among 7 logical feature groups.
   - **Hyperparameter Optimization:** Tuning 8 critical regularization, capacity, and imbalance-weighting hyperparameters of XGBoost.
3. Strict **data leakage prevention** was enforced by stripping direct failure modes (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) and identifiers (`UDI`, `Product ID`).
4. Optimization was driven by **validation Average Precision (PR-AUC)** rather than misleading accuracy.

---

## 2. Preprocessing & Experimental Setup
- **Dataset Size:** 10,000 observations (9,661 normal operations, 339 failure events).
- **Partitioning:** Stratified 70% Train (7,000), 15% Validation (1,500), 15% Isolated Test (1,500).
- **Feature Engineering:** `temperature_difference = Process temperature [K] - Air temperature [K]`.
- **Encoding:** `OneHotEncoder` fitted strictly on training data for categorical feature `Type`.
- **Target Isolation:** The test partition was kept completely untouched during feature selection and optimization.

---

## 3. Discovered Optimal Solution via ABC
- **Optimal Validation Average Precision:** `{abc_summary.get('best_validation_ap', 'N/A'):.4f}`
- **Actual Model Evaluations:** `{abc_summary.get('actual_evaluations', 'N/A')}`
- **Cache Hits:** `{abc_summary.get('cache_hits', 'N/A')}`
- **Selected Feature Groups ({len(selected_features)}/7):**
{chr(10).join([f"  - {feat}" for feat in selected_features])}

### Optimal Hyperparameters:
- `n_estimators`: {best_params.get('n_estimators')}
- `max_depth`: {best_params.get('max_depth')}
- `learning_rate`: {best_params.get('learning_rate')}
- `min_child_weight`: {best_params.get('min_child_weight')}
- `subsample`: {best_params.get('subsample')}
- `colsample_bytree`: {best_params.get('colsample_bytree')}
- `gamma`: {best_params.get('gamma')}
- `scale_pos_weight`: {best_params.get('scale_pos_weight')}

---

## 4. Final Performance Comparison Across Multiple Seeds

{master_table_content}

---

## 5. Academic Discussion & Answers to Key Research Questions

1. **Why is XGBoost appropriate for predictive maintenance?**  
   XGBoost handles mixed tabular data, non-linear sensor thresholds, and missing values natively, while providing gradient-boosted decision boundaries that are resilient to feature scale variations.

2. **Why is accuracy insufficient as the only metric?**  
   With a 96.6% majority negative class, a naive classifier predicting zero failures achieves 96.6% accuracy but provides zero utility for maintenance planning. Precision-Recall AUC (Average Precision) and F1-score evaluate genuine minority failure detection.

3. **How does ABC explore the hyperparameter search space?**  
   Employed bees exploit known local neighborhoods via perturbation ($v_{{ij}} = x_{{ij}} + \\phi_{{ij}}(x_{{ij}} - x_{{kj}})$), Onlooker bees focus search around high-nectar (high-AP) solutions via roulette-wheel selection, and Scout bees abandon stagnant basins when unimproved for $\\ge 5$ trials to maintain global exploration.

4. **How does feature selection influence the model?**  
   Pruning noisy or redundant sensor measurements prevents tree overfitting and enhances generalization on rare failure signatures.

5. **Limitations:**  
   The AI4I 2020 dataset is synthetic (based on physical simulator equations). While it mirrors realistic sensor dynamics, real factory machinery exhibits unmodeled environmental noise, sensor degradation, and non-stationary operating cycles.

---

## 6. Generated Visualizations & Tables Summary
All 15 figures are stored in `results/figures/`:
- `fig01_target_distribution.png`: Class imbalance visualization.
- `fig02_correlation_heatmap.png`: Sensor correlation heatmap.
- `fig03_feature_distributions.png`: Density distributions by failure state.
- `fig04_failure_by_type.png`: Failure breakdown across product categories.
- `fig05_abc_convergence.png`: Best and population mean AP convergence.
- `fig06_best_fitness_vs_evals.png`: Fitness progression vs. evaluation count.
- `fig07_exploration_exploitation.png`: Employed improvements vs. scout resets.
- `fig08_metric_comparison.png`: Bar chart comparison with standard deviations.
- `fig09_baseline_confusion_matrix.png` & `fig10_optimized_confusion_matrix.png`: Test confusion matrices.
- `fig11_roc_curves.png` & `fig12_precision_recall_curves.png`: Discrimination curves.
- `fig13_feature_importance.png`: Gain-based feature importance of ABC model.
- `fig14_feature_selection_frequency.png`: Feature selection frequency across evaluations.
- `fig15_search_analysis.png`: Hyperparameter search landscape distribution.
"""
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Summary report written to {summary_path}")

    def run_all(self, resume: bool = False) -> None:
        """Run complete experimental pipeline."""
        logger.info("Starting COMPLETE experimental pipeline...")
        self.run_stage_eda()
        self.run_stage_baseline()
        self.run_stage_search(resume=resume)
        self.run_stage_final()
        self.run_stage_plots()
        logger.info("Complete pipeline finished successfully!")

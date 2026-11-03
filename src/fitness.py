"""
Fitness evaluation with caching, persistent checkpointing, and evaluation budget enforcement.
Optimization objective: Maximize validation Average Precision (PR-AUC).
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.model import build_xgb_classifier, train_xgb_model
from src.preprocessing import Preprocessor
from src.search_space import SearchSpace
from src.utils import load_json, save_json, setup_logger

logger = setup_logger("fitness")


class FitnessEvaluator:
    """
    Evaluates ABC candidates against validation data using Average Precision (PR-AUC).
    Caches results per unique decoded configuration to prevent redundant training.
    """

    def __init__(
        self,
        X_train_full: pd.DataFrame,
        y_train: pd.Series,
        X_val_full: pd.DataFrame,
        y_val: pd.Series,
        preprocessor: Preprocessor,
        search_space: SearchSpace,
        max_evals: int = 240,
        cache_path: str = "checkpoints/fitness_cache.json",
        eval_log_path: str = "results/logs/abc_evaluations.csv",
        early_stopping_rounds: int = 30,
        random_seed: int = 42,
    ):
        self.X_train_full = X_train_full
        self.y_train = y_train
        self.X_val_full = X_val_full
        self.y_val = y_val
        self.preprocessor = preprocessor
        self.search_space = search_space
        self.max_evals = max_evals
        self.cache_path = cache_path
        self.eval_log_path = eval_log_path
        self.early_stopping_rounds = early_stopping_rounds
        self.random_seed = random_seed

        # Tracking state
        self.cache: Dict[str, float] = {}
        self.eval_counter = 0
        self.cache_hits = 0
        self.total_requests = 0

        # Load existing cache if available (for checkpoint resume)
        self._load_cache()

    def _load_cache(self) -> None:
        """Load persistent cache from disk if it exists."""
        if os.path.exists(self.cache_path):
            try:
                self.cache = load_json(self.cache_path)
                logger.info(f"Loaded {len(self.cache)} cached fitness evaluations from {self.cache_path}")
            except Exception as e:
                logger.warning(f"Could not load cache: {e}. Initializing fresh cache.")
                self.cache = {}

        if os.path.exists(self.eval_log_path):
            try:
                df_existing = pd.read_csv(self.eval_log_path)
                # Count actual model evaluations (cached=False)
                actual_evals = len(df_existing[df_existing["cached"] == False])
                self.eval_counter = actual_evals
                self.total_requests = len(df_existing)
                self.cache_hits = len(df_existing[df_existing["cached"] == True])
                logger.info(
                    f"Resuming evaluator: {self.eval_counter} actual evaluations, "
                    f"{self.cache_hits} cache hits from {self.eval_log_path}"
                )
            except Exception as e:
                logger.warning(f"Could not parse existing evaluation log: {e}")

    def save_cache(self) -> None:
        """Atomically persist cache to disk."""
        save_json(self.cache, self.cache_path)

    def evaluate(
        self,
        x: np.ndarray,
        cycle: int = 0,
        bee_type: str = "employed",
        bee_id: int = 0,
    ) -> Tuple[float, bool]:
        """
        Evaluate candidate solution vector x in [0, 1]^15.

        Args:
            x: Candidate vector.
            cycle: Current optimization cycle.
            bee_type: Bee classification ('employed', 'onlooker', 'scout').
            bee_id: Index of bee in population.

        Returns:
            Tuple of (validation_ap_score, is_cached).
        """
        self.total_requests += 1
        feature_mask, hyperparams = self.search_space.decode(x)
        cache_key = self.search_space.get_cache_key(feature_mask, hyperparams)

        # Check cache
        if cache_key in self.cache:
            self.cache_hits += 1
            ap_score = self.cache[cache_key]
            self._log_evaluation(
                cycle=cycle,
                bee_type=bee_type,
                bee_id=bee_id,
                feature_mask=feature_mask,
                hyperparams=hyperparams,
                ap_score=ap_score,
                cached=True,
                duration_s=0.0,
            )
            return ap_score, True

        # Check evaluation budget
        if self.eval_counter >= self.max_evals:
            logger.warning(f"Evaluation budget of {self.max_evals} reached. Returning last known/default score.")
            # If out of budget, return whatever minimal score or best cached score
            return 0.0, False

        self.eval_counter += 1
        eval_id = self.eval_counter

        # Filter feature columns
        selected_cols = self.preprocessor.get_selected_columns(feature_mask)
        X_tr = self.X_train_full[selected_cols]
        X_va = self.X_val_full[selected_cols]

        # Build and train XGBoost
        model = build_xgb_classifier(
            params=hyperparams,
            random_state=self.random_seed,
            early_stopping_rounds=self.early_stopping_rounds,
        )

        start_time = time.perf_counter()
        train_xgb_model(
            model=model,
            X_train=X_tr,
            y_train=self.y_train,
            eval_set=[(X_va, self.y_val)],
            verbose=False,
        )
        duration_s = time.perf_counter() - start_time

        # Predict validation probabilities and compute Average Precision
        val_probs = model.predict_proba(X_va)[:, 1]
        ap_score = float(average_precision_score(self.y_val, val_probs))

        # Store in cache
        self.cache[cache_key] = ap_score
        self.save_cache()

        # Log to CSV
        self._log_evaluation(
            cycle=cycle,
            bee_type=bee_type,
            bee_id=bee_id,
            feature_mask=feature_mask,
            hyperparams=hyperparams,
            ap_score=ap_score,
            cached=False,
            duration_s=duration_s,
        )

        if eval_id % 10 == 0 or eval_id == 1:
            logger.info(
                f"[Eval #{eval_id}/{self.max_evals}] Cycle={cycle} ({bee_type}_{bee_id}) | "
                f"Features={sum(feature_mask)}/7 | Val AP={ap_score:.4f} | Time={duration_s:.2f}s"
            )

        return ap_score, False

    def _log_evaluation(
        self,
        cycle: int,
        bee_type: str,
        bee_id: int,
        feature_mask: List[bool],
        hyperparams: Dict[str, Any],
        ap_score: float,
        cached: bool,
        duration_s: float,
    ) -> None:
        """Append evaluation result to CSV log."""
        os.makedirs(os.path.dirname(self.eval_log_path), exist_ok=True)
        file_exists = os.path.exists(self.eval_log_path)

        selected_groups = [
            group_name
            for group_idx, (group_name, selected) in enumerate(
                zip(
                    [
                        "Type",
                        "Air_temp",
                        "Process_temp",
                        "Rotational_speed",
                        "Torque",
                        "Tool_wear",
                        "temp_diff",
                    ],
                    feature_mask,
                )
            )
            if selected
        ]

        row = {
            "eval_id": self.eval_counter if not cached else "cached",
            "total_request": self.total_requests,
            "cycle": cycle,
            "bee_type": bee_type,
            "bee_id": bee_id,
            "cached": cached,
            "validation_ap": round(ap_score, 6),
            "duration_s": round(duration_s, 4),
            "num_features_selected": sum(feature_mask),
            "selected_features": "|".join(selected_groups),
            "n_estimators": hyperparams["n_estimators"],
            "max_depth": hyperparams["max_depth"],
            "learning_rate": hyperparams["learning_rate"],
            "min_child_weight": hyperparams["min_child_weight"],
            "subsample": hyperparams["subsample"],
            "colsample_bytree": hyperparams["colsample_bytree"],
            "gamma": hyperparams["gamma"],
            "scale_pos_weight": hyperparams["scale_pos_weight"],
        }

        df_row = pd.DataFrame([row])
        df_row.to_csv(self.eval_log_path, mode="a", index=False, header=not file_exists)

    @staticmethod
    def ap_to_fitness(ap_score: float) -> float:
        """
        Convert Average Precision [0, 1] into strictly positive fitness for roulette selection.
        """
        return max(float(ap_score), 1e-6)


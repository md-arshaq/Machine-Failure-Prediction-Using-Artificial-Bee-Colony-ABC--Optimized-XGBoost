"""
Search space encoding, decoding, and boundary repair.
Encodes 15 continuous variables: 7 feature selection decisions + 8 XGBoost hyperparameters.
"""

import math
from typing import Any, Dict, List, Tuple

import numpy as np


class SearchSpace:
    """
    Manages continuous vector representation x in [0, 1]^15,
    decoding into logical feature selection masks and XGBoost hyperparameters.
    """

    DIMENSION = 15
    NUM_FEATURES = 7
    NUM_HYPERPARAMS = 8

    def __init__(self, bounds_config: Dict[str, Any], n_estimators_cap: int = 600):
        self.bounds = bounds_config
        self.n_estimators_cap = n_estimators_cap

    @staticmethod
    def clip_vector(x: np.ndarray) -> np.ndarray:
        """
        Clip all dimensions to valid continuous domain [0.0, 1.0].
        """
        return np.clip(x, 0.0, 1.0)

    def decode(self, x: np.ndarray) -> Tuple[List[bool], Dict[str, Any]]:
        """
        Decode a 15-dimensional vector into (feature_mask, hyperparams).

        Args:
            x: Continuous numpy array of length 15 in [0, 1].

        Returns:
            Tuple of:
              - feature_mask: List of 7 booleans.
              - hyperparameters: Dict of decoded XGBoost parameters.
        """
        x_clipped = self.clip_vector(x)

        # 1. Feature selection decoding (dimensions 0..6)
        raw_mask = (x_clipped[: self.NUM_FEATURES] >= 0.5).tolist()

        # Fallback repair: guarantee at least one feature is selected
        if not any(raw_mask):
            best_feat_idx = int(np.argmax(x_clipped[: self.NUM_FEATURES]))
            raw_mask[best_feat_idx] = True

        feature_mask = raw_mask

        # 2. Hyperparameter decoding (dimensions 7..14)
        # 7: n_estimators [100, 600]
        n_est_min, n_est_max = self.bounds.get("n_estimators", [100, 600])
        n_est_max = min(n_est_max, self.n_estimators_cap)
        n_estimators = int(round(n_est_min + x_clipped[7] * (n_est_max - n_est_min)))

        # 8: max_depth [2, 10]
        md_min, md_max = self.bounds.get("max_depth", [2, 10])
        max_depth = int(round(md_min + x_clipped[8] * (md_max - md_min)))

        # 9: learning_rate [0.01, 0.30] - Logarithmic scale
        lr_min, lr_max = self.bounds.get("learning_rate", [0.01, 0.30])
        log_lr_min = math.log10(lr_min)
        log_lr_max = math.log10(lr_max)
        learning_rate = round(float(10 ** (log_lr_min + x_clipped[9] * (log_lr_max - log_lr_min))), 4)

        # 10: min_child_weight [1.0, 10.0]
        mcw_min, mcw_max = self.bounds.get("min_child_weight", [1.0, 10.0])
        min_child_weight = round(float(mcw_min + x_clipped[10] * (mcw_max - mcw_min)), 2)

        # 11: subsample [0.60, 1.00]
        sub_min, sub_max = self.bounds.get("subsample", [0.60, 1.00])
        subsample = round(float(sub_min + x_clipped[11] * (sub_max - sub_min)), 3)

        # 12: colsample_bytree [0.60, 1.00]
        col_min, col_max = self.bounds.get("colsample_bytree", [0.60, 1.00])
        colsample_bytree = round(float(col_min + x_clipped[12] * (col_max - col_min)), 3)

        # 13: gamma [0.0, 5.0]
        gam_min, gam_max = self.bounds.get("gamma", [0.0, 5.0])
        gamma = round(float(gam_min + x_clipped[13] * (gam_max - gam_min)), 2)

        # 14: scale_pos_weight [1.0, 30.0]
        spw_min, spw_max = self.bounds.get("scale_pos_weight", [1.0, 30.0])
        scale_pos_weight = round(float(spw_min + x_clipped[14] * (spw_max - spw_min)), 2)

        hyperparams = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "gamma": gamma,
            "scale_pos_weight": scale_pos_weight,
            "tree_method": "hist",
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
        }

        return feature_mask, hyperparams

    @staticmethod
    def get_cache_key(feature_mask: List[bool], hyperparams: Dict[str, Any]) -> str:
        """
        Generate a unique, canonical string key for fitness caching.
        """
        # Exclude non-functional training flags
        filtered_items = [
            (k, v)
            for k, v in hyperparams.items()
            if k not in ["tree_method", "objective", "eval_metric", "random_state", "n_jobs"]
        ]
        sorted_params = tuple(sorted(filtered_items))
        mask_tuple = tuple(bool(b) for b in feature_mask)
        return str((mask_tuple, sorted_params))


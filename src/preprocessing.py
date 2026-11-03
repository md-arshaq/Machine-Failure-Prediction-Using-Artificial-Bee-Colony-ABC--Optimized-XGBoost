"""
Data preprocessing, stratified splitting, and categorical encoding.
Ensures zero data leakage between train, validation, and test splits.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.utils import setup_logger

logger = setup_logger("preprocessing")

LOGICAL_FEATURE_GROUPS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "temperature_difference",
]


class Preprocessor:
    """
    Handles stratified dataset splitting, categorical one-hot encoding,
    and mapping between logical feature groups and physical encoded column names.
    """

    def __init__(self, target_col: str = "Machine failure", random_seed: int = 42):
        self.target_col = target_col
        self.random_seed = random_seed
        self.encoder: OneHotEncoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )
        self.fitted = False
        self.feature_group_mapping: Dict[int, List[str]] = {}
        self.all_feature_cols: List[str] = []

    def stratified_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame into stratified Train (70%), Validation (15%), and Test (15%) subsets.

        Args:
            df: Input cleaned DataFrame.
            train_ratio: Proportion for training.
            val_ratio: Proportion for validation.
            test_ratio: Proportion for testing.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

        y = df[self.target_col]

        # First split into Train (70%) and Temp (30%)
        temp_ratio = val_ratio + test_ratio
        train_df, temp_df = train_test_split(
            df,
            test_size=temp_ratio,
            stratify=y,
            random_state=self.random_seed,
        )

        # Split Temp into Validation and Test (50/50 of temp = 15% and 15% of total)
        val_share_of_temp = val_ratio / temp_ratio
        y_temp = temp_df[self.target_col]
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1.0 - val_share_of_temp),
            stratify=y_temp,
            random_state=self.random_seed,
        )

        logger.info(
            f"Dataset split: Train={len(train_df)} ({len(train_df)/len(df)*100:.1f}%), "
            f"Val={len(val_df)} ({len(val_df)/len(df)*100:.1f}%), "
            f"Test={len(test_df)} ({len(test_df)/len(df)*100:.1f}%)."
        )

        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    @staticmethod
    def sanitize_feature_name(name: str) -> str:
        """
        Clean feature names to satisfy XGBoost naming constraints (no [, ], or <).
        """
        return (
            name.replace("[", "")
            .replace("]", "")
            .replace("<", "")
            .replace(">", "")
            .replace(" ", "_")
        )

    def fit_transform(
        self, train_df: pd.DataFrame, categorical_col: str = "Type"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit OneHotEncoder on training data and transform predictors.

        Args:
            train_df: Training DataFrame including target.
            categorical_col: Name of categorical column to encode.

        Returns:
            Tuple of (X_train DataFrame, y_train Series).
        """
        y_train = train_df[self.target_col].copy()
        X_raw = train_df.drop(columns=[self.target_col])

        # Fit encoder strictly on training categorical data
        encoded_cats = self.encoder.fit_transform(X_raw[[categorical_col]])
        cat_feature_names = self.encoder.get_feature_names_out([categorical_col]).tolist()
        df_encoded_cats = pd.DataFrame(
            encoded_cats, columns=cat_feature_names, index=X_raw.index
        )

        # Combine encoded categories with numerical features
        num_features = [col for col in X_raw.columns if col != categorical_col]
        X_train = pd.concat([df_encoded_cats, X_raw[num_features]], axis=1)

        # Sanitize column names for XGBoost
        clean_col_names = [self.sanitize_feature_name(c) for c in X_train.columns]
        X_train.columns = clean_col_names

        # Build logical group index mapping
        self.feature_group_mapping = {
            0: [self.sanitize_feature_name(c) for c in cat_feature_names],  # Type group (all dummies)
            1: [self.sanitize_feature_name("Air temperature [K]")],
            2: [self.sanitize_feature_name("Process temperature [K]")],
            3: [self.sanitize_feature_name("Rotational speed [rpm]")],
            4: [self.sanitize_feature_name("Torque [Nm]")],
            5: [self.sanitize_feature_name("Tool wear [min]")],
            6: [self.sanitize_feature_name("temperature_difference")],
        }
        self.all_feature_cols = X_train.columns.tolist()
        self.fitted = True

        logger.info(
            f"OneHotEncoder fitted. Total encoded columns: {len(self.all_feature_cols)}. "
            f"Logical groups: {len(self.feature_group_mapping)}."
        )

        return X_train, y_train

    def transform(
        self, df: pd.DataFrame, categorical_col: str = "Type"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Transform validation or test data using fitted encoder.

        Args:
            df: DataFrame to transform.
            categorical_col: Name of categorical column.

        Returns:
            Tuple of (X_transformed DataFrame, y Series).
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before transforming.")

        y = df[self.target_col].copy()
        X_raw = df.drop(columns=[self.target_col])

        encoded_cats = self.encoder.transform(X_raw[[categorical_col]])
        cat_feature_names = self.encoder.get_feature_names_out([categorical_col]).tolist()
        df_encoded_cats = pd.DataFrame(
            encoded_cats, columns=cat_feature_names, index=X_raw.index
        )

        num_features = [col for col in X_raw.columns if col != categorical_col]
        X_trans = pd.concat([df_encoded_cats, X_raw[num_features]], axis=1)
        X_trans.columns = [self.sanitize_feature_name(c) for c in X_trans.columns]

        # Ensure identical column ordering
        X_trans = X_trans[self.all_feature_cols]

        return X_trans, y

    def get_selected_columns(self, feature_mask: List[bool]) -> List[str]:
        """
        Convert a 7-element boolean mask into physical column names.

        Args:
            feature_mask: List of 7 boolean values for logical groups.

        Returns:
            List of matching DataFrame column names.
        """
        if len(feature_mask) != len(self.feature_group_mapping):
            raise ValueError(
                f"Feature mask length {len(feature_mask)} != expected {len(self.feature_group_mapping)}"
            )

        cols: List[str] = []
        for group_idx, is_selected in enumerate(feature_mask):
            if is_selected:
                cols.extend(self.feature_group_mapping[group_idx])
        return cols

    @staticmethod
    def calculate_scale_pos_weight(y: pd.Series) -> float:
        """
        Calculate scale_pos_weight = N_negative / N_positive for imbalanced XGBoost.

        Args:
            y: Binary target series.

        Returns:
            Calculated ratio.
        """
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        if n_pos == 0:
            return 1.0
        return float(n_neg / n_pos)

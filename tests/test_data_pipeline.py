"""
Unit tests for data loading, leakage prevention, and preprocessing.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
from src.data_loader import load_raw_data
from src.feature_engineering import add_engineered_features
from src.preprocessing import Preprocessor, LOGICAL_FEATURE_GROUPS
from src.utils import load_yaml


def test_data_pipeline():
    config = load_yaml("config.yaml")
    ds_cfg = config["dataset"]

    # 1. Load data
    df_clean, meta = load_raw_data(
        filepath=ds_cfg["raw_path"],
        identifier_cols=ds_cfg["identifier_columns"],
        leakage_cols=ds_cfg["leakage_columns"],
        target_col=ds_cfg["target_column"],
    )

    # Assert leakage columns strictly absent
    for col in ds_cfg["leakage_columns"] + ds_cfg["identifier_columns"]:
        assert col not in df_clean.columns, f"Leakage column {col} was found in cleaned data!"

    # Assert target present
    assert ds_cfg["target_column"] in df_clean.columns

    # 2. Feature engineering
    df_fe = add_engineered_features(df_clean)
    assert "temperature_difference" in df_fe.columns
    # Check mathematical correctness
    diff_check = df_fe["Process temperature [K]"] - df_fe["Air temperature [K]"]
    assert (df_fe["temperature_difference"] - diff_check).abs().max() < 1e-6

    # 3. Stratified splitting
    prep = Preprocessor(target_col=ds_cfg["target_column"], random_seed=ds_cfg["random_seed"])
    train_df, val_df, test_df = prep.stratified_split(
        df_fe,
        train_ratio=ds_cfg["train_ratio"],
        val_ratio=ds_cfg["val_ratio"],
        test_ratio=ds_cfg["test_ratio"],
    )

    assert len(train_df) == 7000
    assert len(val_df) == 1500
    assert len(test_df) == 1500

    # Verify class proportions across splits
    train_pos_pct = (train_df[ds_cfg["target_column"]].sum() / len(train_df)) * 100
    val_pos_pct = (val_df[ds_cfg["target_column"]].sum() / len(val_df)) * 100
    test_pos_pct = (test_df[ds_cfg["target_column"]].sum() / len(test_df)) * 100

    assert abs(train_pos_pct - val_pos_pct) < 0.2
    assert abs(train_pos_pct - test_pos_pct) < 0.2

    # 4. Encoding
    X_train, y_train = prep.fit_transform(train_df, categorical_col="Type")
    X_val, y_val = prep.transform(val_df, categorical_col="Type")
    X_test, y_test = prep.transform(test_df, categorical_col="Type")

    assert X_train.shape[1] == 9  # 3 dummies for Type (H, L, M) + 5 numerical + 1 engineered
    assert X_val.shape[1] == 9
    assert X_test.shape[1] == 9

    # 5. Logical group mapping
    assert len(prep.feature_group_mapping) == 7
    all_selected = prep.get_selected_columns([True] * 7)
    assert len(all_selected) == 9

    # Test Type group selection: if Type group is False, no Type dummies selected
    no_type = prep.get_selected_columns([False, True, True, True, True, True, True])
    assert "Type_H" not in no_type
    assert "Type_L" not in no_type
    assert "Type_M" not in no_type
    assert len(no_type) == 6

    # 6. scale_pos_weight
    spw = prep.calculate_scale_pos_weight(y_train)
    assert spw > 20.0  # ~28.5

    print("\n[SUCCESS] Data pipeline unit tests passed completely!")


if __name__ == "__main__":
    test_data_pipeline()

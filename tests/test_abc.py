"""
Unit tests for ABC optimization operators, search space decoding, boundary clipping,
and caching logic.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from src.search_space import SearchSpace
from src.utils import load_yaml


def test_search_space_decoding():
    config = load_yaml("config.yaml")
    bounds = config["xgboost"]["search_bounds"]
    ss = SearchSpace(bounds_config=bounds)

    # Test 1: Vector with all zeros (should trigger fallback repair for features)
    x_zero = np.zeros(15)
    mask_zero, params_zero = ss.decode(x_zero)
    assert any(mask_zero), "Fallback repair failed: at least one feature must be selected!"
    assert sum(mask_zero) == 1
    assert params_zero["n_estimators"] == 100
    assert params_zero["max_depth"] == 2
    assert np.isclose(params_zero["learning_rate"], 0.01)
    assert np.isclose(params_zero["scale_pos_weight"], 1.0)

    # Test 2: Vector with all ones
    x_one = np.ones(15)
    mask_one, params_one = ss.decode(x_one)
    assert all(mask_one), "All features should be selected when values are 1.0"
    assert sum(mask_one) == 7
    assert params_one["n_estimators"] == 600
    assert params_one["max_depth"] == 10
    assert np.isclose(params_one["learning_rate"], 0.30)
    assert np.isclose(params_one["scale_pos_weight"], 30.0)

    # Test 3: Cache key generation
    key1 = ss.get_cache_key(mask_zero, params_zero)
    key2 = ss.get_cache_key(mask_zero, params_zero)
    key3 = ss.get_cache_key(mask_one, params_one)
    assert key1 == key2
    assert key1 != key3

    # Test 4: Boundary clipping
    x_out_of_bounds = np.array([-0.5] * 15)
    x_clipped = ss.clip_vector(x_out_of_bounds)
    assert np.all(x_clipped >= 0.0)
    assert np.all(x_clipped <= 1.0)

    print("\n[SUCCESS] SearchSpace unit tests passed!")


def test_abc_operators():
    from src.abc_optimizer import ArtificialBeeColony

    # Mock evaluator for fast algorithm testing
    class MockEvaluator:
        def __init__(self):
            self.eval_counter = 0
            self.cache_hits = 0
            self.max_evals = 50

        def evaluate(self, x, cycle=0, bee_type="employed", bee_id=0):
            self.eval_counter += 1
            # Sphere-like objective function for testing
            score = float(1.0 / (1.0 + np.sum(x ** 2)))
            return score, False

    evaluator = MockEvaluator()
    abc = ArtificialBeeColony(
        evaluator=evaluator,
        n_food_sources=5,
        n_onlooker=5,
        max_cycles=3,
        abandonment_limit=2,
        random_seed=42,
        checkpoint_path="checkpoints/test_abc_state.json",
        convergence_path="results/logs/test_convergence.csv",
    )

    best_sol, best_fit, history = abc.optimize()

    assert best_sol is not None
    assert len(best_sol) == 15
    assert best_fit > 0.0
    assert len(history) == 3
    assert os.path.exists("checkpoints/test_abc_state.json")

    # Clean up test artifacts
    if os.path.exists("checkpoints/test_abc_state.json"):
        os.remove("checkpoints/test_abc_state.json")
    if os.path.exists("results/logs/test_convergence.csv"):
        os.remove("results/logs/test_convergence.csv")

    print("[SUCCESS] ABC optimizer operators unit tests passed!")


if __name__ == "__main__":
    test_search_space_decoding()
    test_abc_operators()


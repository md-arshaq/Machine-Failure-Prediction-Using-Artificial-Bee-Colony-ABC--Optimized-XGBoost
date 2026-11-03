"""
Artificial Bee Colony (ABC) optimizer implemented from scratch in pure NumPy.
Jointly optimizes continuous feature-selection decisions and XGBoost hyperparameters.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.fitness import FitnessEvaluator
from src.utils import load_json, save_json, setup_logger

logger = setup_logger("abc_optimizer")


class ArtificialBeeColony:
    """
    Artificial Bee Colony (ABC) metaheuristic algorithm.
    Models Employed Bees, Onlooker Bees (roulette-wheel selection), and Scout Bees (abandonment).
    """

    DIMENSION = 15

    def __init__(
        self,
        evaluator: FitnessEvaluator,
        n_food_sources: int = 10,
        n_onlooker: int = 10,
        max_cycles: int = 10,
        abandonment_limit: int = 5,
        random_seed: int = 42,
        checkpoint_path: str = "checkpoints/abc_state.json",
        convergence_path: str = "results/logs/abc_convergence.csv",
    ):
        self.evaluator = evaluator
        self.n_food_sources = n_food_sources
        self.n_onlooker = n_onlooker
        self.max_cycles = max_cycles
        self.abandonment_limit = abandonment_limit
        self.random_seed = random_seed
        self.checkpoint_path = checkpoint_path
        self.convergence_path = convergence_path

        self.rng = np.random.default_rng(random_seed)

        # Internal population state
        self.population = np.zeros((self.n_food_sources, self.DIMENSION), dtype=np.float64)
        self.fitness_values = np.zeros(self.n_food_sources, dtype=np.float64)
        self.trial_counters = np.zeros(self.n_food_sources, dtype=np.int32)

        self.best_solution: Optional[np.ndarray] = None
        self.best_fitness: float = -1.0
        self.cycle_history: List[Dict[str, Any]] = []
        self.start_cycle: int = 0

    def initialize_population(self) -> None:
        """
        Initialize food sources uniformly in [0, 1]^15 and evaluate initial fitness.
        """
        logger.info(
            f"Initializing ABC population with {self.n_food_sources} food sources in [0, 1]^{self.DIMENSION}..."
        )
        self.population = self.rng.uniform(0.0, 1.0, size=(self.n_food_sources, self.DIMENSION))
        self.trial_counters = np.zeros(self.n_food_sources, dtype=np.int32)

        for i in range(self.n_food_sources):
            ap_score, _ = self.evaluator.evaluate(
                x=self.population[i],
                cycle=0,
                bee_type="init",
                bee_id=i,
            )
            self.fitness_values[i] = ap_score
            if ap_score > self.best_fitness:
                self.best_fitness = ap_score
                self.best_solution = self.population[i].copy()

        logger.info(f"Initialization complete. Initial Best Val AP: {self.best_fitness:.4f}")

    def save_checkpoint(self, cycle: int) -> None:
        """Atomically persist complete optimizer state."""
        state = {
            "cycle": cycle,
            "population": self.population.tolist(),
            "fitness_values": self.fitness_values.tolist(),
            "trial_counters": self.trial_counters.tolist(),
            "best_solution": self.best_solution.tolist() if self.best_solution is not None else None,
            "best_fitness": float(self.best_fitness),
            "cycle_history": self.cycle_history,
        }
        save_json(state, self.checkpoint_path)
        logger.info(f"Cycle {cycle} checkpoint saved to {self.checkpoint_path}")

    def load_checkpoint(self) -> bool:
        """
        Reload state from checkpoint file if present.
        Returns True if successful resume, False otherwise.
        """
        if not os.path.exists(self.checkpoint_path):
            return False

        try:
            state = load_json(self.checkpoint_path)
            self.start_cycle = state["cycle"] + 1
            self.population = np.array(state["population"], dtype=np.float64)
            self.fitness_values = np.array(state["fitness_values"], dtype=np.float64)
            self.trial_counters = np.array(state["trial_counters"], dtype=np.int32)
            self.best_solution = np.array(state["best_solution"], dtype=np.float64)
            self.best_fitness = float(state["best_fitness"])
            self.cycle_history = state.get("cycle_history", [])
            logger.info(
                f"Resumed ABC state from cycle {state['cycle']}. Best fitness: {self.best_fitness:.4f}"
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to load checkpoint from {self.checkpoint_path}: {e}")
            return False

    def _roulette_wheel_selection(self) -> int:
        """
        Select food source index using roulette-wheel selection based on fitness.
        """
        # Ensure positive fitness values
        pos_fitness = np.maximum(self.fitness_values, 1e-6)
        total_fit = np.sum(pos_fitness)
        probabilities = pos_fitness / total_fit
        return int(self.rng.choice(self.n_food_sources, p=probabilities))

    def _generate_candidate(self, i: int) -> np.ndarray:
        """
        Generate candidate solution v_i in neighborhood of x_i:
        v_ij = x_ij + phi_ij * (x_ij - x_kj)
        """
        # Select neighbor k != i
        neighbor_indices = [idx for idx in range(self.n_food_sources) if idx != i]
        k = int(self.rng.choice(neighbor_indices))

        # Select dimension j to perturb (Karaboga standard)
        j = int(self.rng.integers(0, self.DIMENSION))

        # Perturbation parameter phi in [-1, 1]
        phi = self.rng.uniform(-1.0, 1.0)

        candidate = self.population[i].copy()
        candidate[j] = candidate[j] + phi * (candidate[j] - self.population[k, j])

        # Clip to valid bounds [0, 1]
        return np.clip(candidate, 0.0, 1.0)

    def optimize(self, resume: bool = False) -> Tuple[np.ndarray, float, List[Dict[str, Any]]]:
        """
        Execute full ABC optimization loop.

        Args:
            resume: If True, attempt to load existing checkpoint.

        Returns:
            Tuple of (best_solution_vector, best_fitness, cycle_history).
        """
        resumed = False
        if resume:
            resumed = self.load_checkpoint()

        if not resumed:
            self.initialize_population()
            self.start_cycle = 1

        logger.info(
            f"Starting ABC optimization from cycle {self.start_cycle} to {self.max_cycles} "
            f"(Max Evals: {self.evaluator.max_evals})..."
        )

        for cycle in range(self.start_cycle, self.max_cycles + 1):
            improvements = 0
            scout_resets = 0

            # --- Phase 1: Employed Bees ---
            for i in range(self.n_food_sources):
                if self.evaluator.eval_counter >= self.evaluator.max_evals:
                    break

                candidate = self._generate_candidate(i)
                cand_fitness, _ = self.evaluator.evaluate(
                    x=candidate,
                    cycle=cycle,
                    bee_type="employed",
                    bee_id=i,
                )

                # Greedy selection
                if cand_fitness > self.fitness_values[i]:
                    self.population[i] = candidate
                    self.fitness_values[i] = cand_fitness
                    self.trial_counters[i] = 0
                    improvements += 1
                    if cand_fitness > self.best_fitness:
                        self.best_fitness = cand_fitness
                        self.best_solution = candidate.copy()
                else:
                    self.trial_counters[i] += 1

            # --- Phase 2: Onlooker Bees ---
            for o in range(self.n_onlooker):
                if self.evaluator.eval_counter >= self.evaluator.max_evals:
                    break

                # Select food source based on nectar quality
                i = self._roulette_wheel_selection()
                candidate = self._generate_candidate(i)
                cand_fitness, _ = self.evaluator.evaluate(
                    x=candidate,
                    cycle=cycle,
                    bee_type="onlooker",
                    bee_id=o,
                )

                # Greedy selection
                if cand_fitness > self.fitness_values[i]:
                    self.population[i] = candidate
                    self.fitness_values[i] = cand_fitness
                    self.trial_counters[i] = 0
                    improvements += 1
                    if cand_fitness > self.best_fitness:
                        self.best_fitness = cand_fitness
                        self.best_solution = candidate.copy()
                else:
                    self.trial_counters[i] += 1

            # --- Phase 3: Scout Bees ---
            # Identify food source with highest abandonment counter
            max_trial_idx = int(np.argmax(self.trial_counters))
            if self.trial_counters[max_trial_idx] >= self.abandonment_limit:
                if self.evaluator.eval_counter < self.evaluator.max_evals:
                    # Abandon and randomly scout a new solution
                    scout_candidate = self.rng.uniform(0.0, 1.0, size=self.DIMENSION)
                    scout_fitness, _ = self.evaluator.evaluate(
                        x=scout_candidate,
                        cycle=cycle,
                        bee_type="scout",
                        bee_id=max_trial_idx,
                    )
                    self.population[max_trial_idx] = scout_candidate
                    self.fitness_values[max_trial_idx] = scout_fitness
                    self.trial_counters[max_trial_idx] = 0
                    scout_resets += 1

                    if scout_fitness > self.best_fitness:
                        self.best_fitness = scout_fitness
                        self.best_solution = scout_candidate.copy()

            # Record cycle statistics
            mean_fitness = float(np.mean(self.fitness_values))
            cycle_stat = {
                "cycle": cycle,
                "best_so_far_ap": round(self.best_fitness, 6),
                "population_mean_ap": round(mean_fitness, 6),
                "improvements": improvements,
                "scout_resets": scout_resets,
                "actual_evaluations": self.evaluator.eval_counter,
                "cache_hits": self.evaluator.cache_hits,
            }
            self.cycle_history.append(cycle_stat)

            logger.info(
                f"[Cycle {cycle}/{self.max_cycles}] Best AP: {self.best_fitness:.4f} | "
                f"Mean AP: {mean_fitness:.4f} | Improvements: {improvements} | "
                f"Scouts: {scout_resets} | Actual Evals: {self.evaluator.eval_counter}/{self.evaluator.max_evals}"
            )

            # Persist checkpoint after each completed cycle
            self.save_checkpoint(cycle)

            # Check if evaluation budget exhausted
            if self.evaluator.eval_counter >= self.evaluator.max_evals:
                logger.info(f"Evaluation budget limit ({self.evaluator.max_evals}) reached. Stopping ABC search.")
                break

        # Save convergence history to CSV
        os.makedirs(os.path.dirname(self.convergence_path), exist_ok=True)
        import pandas as pd
        df_conv = pd.DataFrame(self.cycle_history)
        df_conv.to_csv(self.convergence_path, index=False)
        logger.info(f"ABC convergence history saved to {self.convergence_path}")

        return self.best_solution, self.best_fitness, self.cycle_history


"""
Command-Line Interface (CLI) Entry Point.
Executes experimental stages: eda, baseline, search, final, plots, all.
Supports --quick for rapid smoke testing and --resume for checkpoint continuation.
"""

import argparse
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.experiment import ExperimentRunner
from src.utils import load_yaml, set_seed, setup_logger

logger = setup_logger("cli")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ABC-Optimized XGBoost for Machine Failure Prediction"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["eda", "baseline", "search", "final", "plots", "all"],
        help="Pipeline stage to execute (eda, baseline, search, final, plots, all).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Execute smoke test with minimal population, cycles, and estimators.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume ABC optimization search from saved checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"Loading configuration from {args.config}...")
    config = load_yaml(args.config)

    # Global seed initialization
    seed = config["dataset"].get("random_seed", 42)
    set_seed(seed)

    runner = ExperimentRunner(config=config, quick_mode=args.quick)

    logger.info(f"Executing stage: '{args.stage}' (Quick={args.quick}, Resume={args.resume})")

    if args.stage == "eda":
        runner.run_stage_eda()
    elif args.stage == "baseline":
        runner.run_stage_baseline()
    elif args.stage == "search":
        runner.run_stage_search(resume=args.resume)
    elif args.stage == "final":
        runner.run_stage_final()
    elif args.stage == "plots":
        runner.run_stage_plots()
    elif args.stage == "all":
        runner.run_all(resume=args.resume)
    else:
        raise ValueError(f"Unknown stage: {args.stage}")

    logger.info("Process finished successfully.")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""CLI entry point to execute RobustRAG experiments."""

import argparse
import sys
from pathlib import Path

# Add src to python path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from robustrag.config import AppConfig
from robustrag.experiment import ExperimentRunner
from robustrag.utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RobustRAG Experiment Runner: Controlled RAG context-noise investigation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML experiment configuration file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute in mock/dry-run mode without downloading models or incurring inference costs.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Override question sample size from configuration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed for sampling and noise generation.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["local", "groq", "mock"],
        default=None,
        help="Override generator model backend ('local', 'groq', or 'mock').",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger("robustrag.cli")
    logger.info("Initializing RobustRAG CLI execution...")

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Configuration file not found: %s", config_path)
        return 1

    try:
        config = AppConfig.from_yaml(config_path)

        # Apply CLI overrides
        if args.dry_run:
            logger.info("Dry-run mode activated: using mock backends and smoke test flag.")
            config.generator.backend = "mock"
            config.reranker.backend = "mock"
            config.is_smoke_test = True

        if args.backend:
            config.generator.backend = args.backend
            if args.backend == "mock":
                config.reranker.backend = "mock"

        if args.sample_size is not None:
            config.dataset.sample_size = args.sample_size

        if args.seed is not None:
            config.random_seed = args.seed

        val_errors = config.validate()
        if val_errors:
            for err in val_errors:
                logger.error("Config validation error: %s", err)
            return 1

        est_calls = config.estimate_generator_calls()
        logger.info(
            "Starting experiment: '%s' (Sample Size: %d, Seed: %d, Est. Gen Calls: %d)",
            config.name, config.dataset.sample_size, config.random_seed, est_calls
        )

        runner = ExperimentRunner(config)
        results = runner.run()

        logger.info("Experiment completed successfully!")
        logger.info("Run ID: %s", results["run_id"])
        logger.info("Outputs saved to: %s", results["run_dir"])
        return 0

    except Exception as e:
        logger.exception("Fatal error during experiment execution: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())

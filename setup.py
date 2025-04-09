"""Experiment manager for running multiple training experiments."""

# pylint: disable=too-many-arguments,too-many-locals,import-outside-toplevel

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console

from ..utils.logging_utils import configure_logging, get_logger
from .train import train_hybrid_model

app = typer.Typer(help="IPSEM 2025 Experiment Manager", add_completion=False)
console = Console()
logger = get_logger(__name__)


@app.command("run")
def run_experiments(
    config_file: str = typer.Option(
        ..., "--config-file", "-c", help="Path to the experiment configuration file"
    ),
    output_dir: str = typer.Option(
        "experiments", "--output-dir", "-o", help="Base directory for experiment outputs"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """Run multiple training experiments based on a configuration file."""
    try:
        # Load experiment configurations
        with open(config_file, "r") as f:
            experiments = json.load(f)

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging
        log_level = "DEBUG" if verbose else "INFO"
        log_file = output_dir / "experiment_manager.log"
        configure_logging(
            level=log_level,
            log_to_console=True,
            log_to_file=True,
            log_file=str(log_file),
        )

        logger.info(f"Loaded {len(experiments)} experiments from {config_file}")
        logger.info(f"Logging to: {log_file}")

        # Run each experiment
        for i, experiment in enumerate(experiments, 1):
            logger.info(f"Starting experiment {i}/{len(experiments)}: {experiment}")

            # Create run directory for this experiment
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"experiment_{i:03d}_{timestamp}"
            run_dir = output_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            # Update experiment parameters with run-specific paths
            experiment["run_dir"] = run_dir
            experiment["stats_file"] = run_dir / "training_stats.csv"
            experiment["model_save_path"] = run_dir / "model_final.pt"

            # Run the training
            result = train_hybrid_model(**experiment)

            # Save experiment result metadata
            result_metadata = {
                "experiment": experiment,
                "result": {
                    "final_stats": result["final_stats"],
                    "test_metrics": result["test_metrics"],
                    "hardware_info": result["hardware_info"],
                },
            }
            with open(run_dir / "result_metadata.json", "w") as f:
                json.dump(result_metadata, f, indent=2)

            logger.info(f"Experiment {i} completed. Results saved to {run_dir}")

        logger.info("All experiments completed successfully")
    except Exception as e:
        logger.exception(f"Experiment manager failed: {e}")
        raise typer.Exit(code=1)


def main():
    """Main entry point for the experiment manager CLI."""
    try:
        return app()
    except Exception as e:
        logger.exception("An error occurred: %s", e)
        return 1


if __name__ == "__main__":
    main()
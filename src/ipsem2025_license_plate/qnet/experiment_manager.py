"""Experiment manager for running multiple quantum model configurations."""

import itertools
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from ..utils.logging_utils import configure_logging, get_logger
from .cli import train_command
from .utils import get_hardware_info

logger = get_logger(__name__)
console = Console()
app = typer.Typer(help="IPSEM 2025 Experiment Manager")


def create_experiment_directory(
    base_dir: str, experiment_name: Optional[str] = None
) -> Path:
    """Create a directory for the experiment with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name:
        dir_name = f"{timestamp}_{experiment_name}"
    else:
        dir_name = timestamp

    experiment_dir = Path(base_dir) / dir_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def run_single_experiment(
    n_qubits: int,
    ansatz_reps: int,
    feature_map_reps: int,
    experiment_dir: Path,
    epochs: int = 10,
    batch_size: int = 128,
    dataset_type: str = "emnist",
    dataset_path: str = "data",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    learning_rate: float = 1e-3,
    preload_data: bool = True,
    num_workers: Optional[int] = None,
    use_gpu_for_qnn: bool = True,
    verbose: bool = False,
) -> Dict:
    """Run a single experiment with the specified parameters."""
    experiment_name = f"q{n_qubits}_a{ansatz_reps}_f{feature_map_reps}"
    run_dir = experiment_dir / experiment_name

    console.print(f"[bold blue]Starting experiment: {experiment_name}[/bold blue]")

    # Configure log file for this specific experiment
    log_file = run_dir / "experiment.log"
    configure_logging(
        level="DEBUG" if verbose else "INFO",
        log_to_console=False,  # Don't log to console for this run to avoid cluttering output
        log_to_file=True,
        log_file=str(log_file),
    )

    logger.info(f"Running experiment {experiment_name} in {run_dir}")

    # Create arguments for the train command
    train_args = {
        "n_qubits": n_qubits,
        "ansatz_reps": ansatz_reps,
        "feature_map_reps": feature_map_reps,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "learning_rate": learning_rate,
        "dataset_type": dataset_type,
        "dataset_path": dataset_path,
        "output_dir": str(run_dir),
        "save_intermediate": True,
        "run_test": True,
        "verbose": verbose,
        "use_gpu_for_qnn": use_gpu_for_qnn,
        "preload_data": preload_data,
        "num_workers": num_workers,
    }

    # Log the experiment parameters
    logger.info(f"Experiment parameters: {train_args}")

    # Run the train command
    try:
        start_time = time.time()
        return_code = train_command(**train_args)
        end_time = time.time()

        # Extract results from the experiment
        results = {
            "experiment_name": experiment_name,
            "n_qubits": n_qubits,
            "ansatz_reps": ansatz_reps,
            "feature_map_reps": feature_map_reps,
            "runtime_seconds": end_time - start_time,
            "success": return_code == 0,
            "run_dir": str(run_dir),
        }

        # Read the final metrics from the saved metadata
        try:
            metadata_path = run_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    results["test_accuracy"] = metadata.get("test_accuracy", None)

            # Read the final epoch stats
            stats_file = run_dir / "training_stats.csv"
            if stats_file.exists():
                df = pd.read_csv(stats_file)
                if not df.empty:
                    last_epoch = df.iloc[-1]
                    results["final_train_loss"] = last_epoch.get("train_loss", None)
                    results["final_train_accuracy"] = last_epoch.get(
                        "train_accuracy", None
                    )
                    results["final_val_loss"] = last_epoch.get("val_loss", None)
                    results["final_val_accuracy"] = last_epoch.get("val_accuracy", None)
                    results["total_time"] = last_epoch.get("total_time", None)
                    results["peak_memory_mb"] = last_epoch.get(
                        "gpu_peak_memory_mb", None
                    )
        except Exception as e:
            logger.error(f"Error extracting results: {e}")

        logger.info(f"Experiment {experiment_name} completed with results: {results}")
        return results

    except Exception as e:
        logger.exception(f"Experiment failed: {e}")
        return {
            "experiment_name": experiment_name,
            "n_qubits": n_qubits,
            "ansatz_reps": ansatz_reps,
            "feature_map_reps": feature_map_reps,
            "success": False,
            "error": str(e),
        }


@app.command("run")
def run_experiments(
    qubits_range: str = typer.Option(
        "6",
        "--qubits",
        "-q",
        help="Comma-separated list or range of qubits to test (e.g., '4,6,8' or '4-8')",
    ),
    ansatz_range: str = typer.Option(
        "1-5",
        "--ansatz",
        "-a",
        help="Comma-separated list or range of ansatz repetitions to test (e.g., '1,2,3' or '1-5')",
    ),
    feature_map_range: str = typer.Option(
        "1-3",
        "--feature-map",
        "-f",
        help="Comma-separated list or range of feature map repetitions to test (e.g., '1,2' or '1-3')",
    ),
    epochs: int = typer.Option(
        10, "--epochs", "-e", help="Number of epochs for each experiment"
    ),
    batch_size: int = typer.Option(
        128, "--batch-size", "-b", help="Batch size for training"
    ),
    output_dir: str = typer.Option(
        "experiments",
        "--output-dir",
        "-o",
        help="Base directory for experiment outputs",
    ),
    experiment_name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Name for this experiment suite"
    ),
    dataset_type: str = typer.Option(
        "emnist", "--dataset-type", "-d", help="Dataset to use"
    ),
    dataset_path: str = typer.Option(
        "data", "--dataset-path", "-p", help="Path to dataset"
    ),
    learning_rate: float = typer.Option(
        1e-3, "--learning-rate", "-l", help="Learning rate"
    ),
    use_gpu: bool = typer.Option(
        True, "--use-gpu/--no-gpu", help="Use GPU for quantum simulation"
    ),
    preload_data: bool = typer.Option(
        True, "--preload-data/--no-preload", help="Preload dataset into memory"
    ),
    skip_existing: bool = typer.Option(
        False, "--skip-existing", help="Skip configurations that have already been run"
    ),
    num_workers: Optional[int] = typer.Option(
        None, "--num-workers", help="Number of dataloader workers (default: auto)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> int:
    """Run a series of experiments with different quantum circuit configurations."""
    try:
        # Parse the ranges
        def parse_range(range_str: str) -> List[int]:
            if "-" in range_str:
                start, end = map(int, range_str.split("-"))
                return list(range(start, end + 1))
            else:
                return [int(x) for x in range_str.split(",")]

        qubits_list = parse_range(qubits_range)
        ansatz_list = parse_range(ansatz_range)
        feature_map_list = parse_range(feature_map_range)

        # Create experiment directory
        experiment_dir = create_experiment_directory(output_dir, experiment_name)

        # Save experiment configuration
        config = {
            "timestamp": datetime.now().isoformat(),
            "experiment_name": experiment_name,
            "qubits_range": qubits_range,
            "ansatz_range": ansatz_range,
            "feature_map_range": feature_map_range,
            "epochs": epochs,
            "batch_size": batch_size,
            "dataset_type": dataset_type,
            "learning_rate": learning_rate,
            "use_gpu": use_gpu,
            "preload_data": preload_data,
            "hardware_info": get_hardware_info(),
        }

        with open(experiment_dir / "experiment_config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Calculate total number of experiments
        total_experiments = len(qubits_list) * len(ansatz_list) * len(feature_map_list)
        console.print(f"[bold]Running {total_experiments} experiments[/bold]")
        console.print(f"Experiment directory: {experiment_dir}")
        console.print(
            f"Configuration: {qubits_list} qubits, {ansatz_list} ansatz reps, {feature_map_list} feature map reps"
        )

        # Generate all combinations
        combinations = list(
            itertools.product(qubits_list, ansatz_list, feature_map_list)
        )
        results = []

        # Create results dataframe and save it
        results_df = pd.DataFrame(
            columns=[
                "experiment_name",
                "n_qubits",
                "ansatz_reps",
                "feature_map_reps",
                "success",
                "test_accuracy",
                "final_train_accuracy",
                "final_val_accuracy",
                "runtime_seconds",
                "total_time",
                "peak_memory_mb",
            ]
        )
        results_path = experiment_dir / "results.csv"
        results_df.to_csv(results_path, index=False)

        # Show progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Running experiments...", total=total_experiments
            )

            for i, (n_qubits, ansatz_reps, feature_map_reps) in enumerate(
                combinations, 1
            ):
                progress.update(
                    task,
                    description=f"[cyan]Experiment {i}/{total_experiments}: q={n_qubits}, a={ansatz_reps}, f={feature_map_reps}",
                )

                # Run the experiment
                result = run_single_experiment(
                    n_qubits=n_qubits,
                    ansatz_reps=ansatz_reps,
                    feature_map_reps=feature_map_reps,
                    experiment_dir=experiment_dir,
                    epochs=epochs,
                    batch_size=batch_size,
                    dataset_type=dataset_type,
                    dataset_path=dataset_path,
                    learning_rate=learning_rate,
                    preload_data=preload_data,
                    num_workers=num_workers,
                    use_gpu_for_qnn=use_gpu,
                    verbose=verbose,
                )

                results.append(result)

                # Update the results CSV after each experiment
                new_row = pd.DataFrame([result])
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                results_df.to_csv(results_path, index=False)

                progress.advance(task)

        # Generate summary report
        console.print("\n[bold green]Experiment Suite Completed![/bold green]")

        # Create results table
        table = Table(title="Experiment Results")
        table.add_column("Qubits", justify="center", style="cyan")
        table.add_column("Ansatz", justify="center", style="magenta")
        table.add_column("Feature Map", justify="center", style="green")
        table.add_column("Test Acc", justify="right", style="yellow")
        table.add_column("Train Acc", justify="right")
        table.add_column("Val Acc", justify="right")
        table.add_column("Runtime", justify="right")

        for result in results:
            if result["success"]:
                table.add_row(
                    str(result["n_qubits"]),
                    str(result["ansatz_reps"]),
                    str(result["feature_map_reps"]),
                    (
                        f"{result.get('test_accuracy', 'N/A'):.2f}%"
                        if result.get("test_accuracy")
                        else "N/A"
                    ),
                    (
                        f"{result.get('final_train_accuracy', 'N/A'):.2f}%"
                        if result.get("final_train_accuracy")
                        else "N/A"
                    ),
                    (
                        f"{result.get('final_val_accuracy', 'N/A'):.2f}%"
                        if result.get("final_val_accuracy")
                        else "N/A"
                    ),
                    f"{result.get('runtime_seconds', 0):.1f}s",
                )
            else:
                table.add_row(
                    str(result["n_qubits"]),
                    str(result["ansatz_reps"]),
                    str(result["feature_map_reps"]),
                    "FAILED",
                    "FAILED",
                    "FAILED",
                    "N/A",
                )

        console.print(table)
        console.print(f"\nDetailed results saved to: {results_path}")
        console.print(f"Experiment data directory: {experiment_dir}")

        # Generate visualization of results if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Filter successful experiments
            successful_df = results_df[results_df["success"] == True].copy()

            if not successful_df.empty:
                # Plot test accuracy by qubit and ansatz
                fig, axs = plt.subplots(1, 2, figsize=(15, 6))

                # Convert accuracy columns to numeric
                for col in [
                    "test_accuracy",
                    "final_train_accuracy",
                    "final_val_accuracy",
                ]:
                    if col in successful_df.columns:
                        successful_df[col] = pd.to_numeric(
                            successful_df[col], errors="coerce"
                        )

                # Plot 1: Accuracy vs Ansatz Depth
                for q in successful_df["n_qubits"].unique():
                    data = successful_df[successful_df["n_qubits"] == q]
                    axs[0].plot(
                        data["ansatz_reps"],
                        data["test_accuracy"],
                        marker="o",
                        label=f"{q} qubits",
                    )

                axs[0].set_xlabel("Ansatz Repetitions")
                axs[0].set_ylabel("Test Accuracy (%)")
                axs[0].set_title("Test Accuracy vs Ansatz Depth")
                axs[0].legend()
                axs[0].grid(True)

                # Plot 2: Accuracy vs Feature Map Depth
                for q in successful_df["n_qubits"].unique():
                    data = successful_df[successful_df["n_qubits"] == q]
                    axs[1].plot(
                        data["feature_map_reps"],
                        data["test_accuracy"],
                        marker="o",
                        label=f"{q} qubits",
                    )

                axs[1].set_xlabel("Feature Map Repetitions")
                axs[1].set_ylabel("Test Accuracy (%)")
                axs[1].set_title("Test Accuracy vs Feature Map Depth")
                axs[1].legend()
                axs[1].grid(True)

                plt.tight_layout()
                plot_path = experiment_dir / "results_plot.png"
                plt.savefig(plot_path)
                console.print(f"Results visualization saved to: {plot_path}")
        except ImportError:
            console.print(
                "[yellow]Matplotlib not available. Skipping results visualization.[/yellow]"
            )
        except Exception as e:
            console.print(f"[yellow]Failed to generate visualization: {e}[/yellow]")

        return 0

    except Exception as e:
        console.print(f"[bold red]Error running experiments: {e}[/bold red]")
        logger.exception("Error in experiment manager")
        return 1


@app.command("analyze")
def analyze_results(
    results_dir: str = typer.Option(
        ..., "--dir", "-d", help="Directory containing experiment results"
    ),
    output_file: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file for the analysis report"
    ),
):
    """Analyze results from a previous experiment run."""
    try:
        results_dir = Path(results_dir)
        results_path = results_dir / "results.csv"

        if not results_path.exists():
            console.print(
                f"[bold red]Results file not found: {results_path}[/bold red]"
            )
            return 1

        # Load results
        results_df = pd.read_csv(results_path)

        # Ensure numeric columns are properly typed
        for col in [
            "test_accuracy",
            "final_train_accuracy",
            "final_val_accuracy",
            "runtime_seconds",
        ]:
            if col in results_df.columns:
                results_df[col] = pd.to_numeric(results_df[col], errors="coerce")

        # Basic statistics
        console.print(f"[bold]Analyzing results from: {results_dir}[/bold]")
        console.print(f"Total experiments: {len(results_df)}")
        console.print(f"Successful experiments: {results_df['success'].sum()}")

        if results_df["success"].sum() > 0:
            successful = results_df[results_df["success"] == True]

            # Find best configuration
            if "test_accuracy" in successful.columns:
                best_idx = successful["test_accuracy"].idxmax()
                best_config = successful.loc[best_idx]

                console.print("\n[bold]Best Configuration:[/bold]")
                console.print(f"  Qubits: {best_config['n_qubits']}")
                console.print(f"  Ansatz repetitions: {best_config['ansatz_reps']}")
                console.print(
                    f"  Feature map repetitions: {best_config['feature_map_reps']}"
                )
                console.print(f"  Test accuracy: {best_config['test_accuracy']:.2f}%")
                console.print(
                    f"  Runtime: {best_config['runtime_seconds']:.1f} seconds"
                )

            # Generate visualizations
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns

                # Plot configuration
                plt.style.use("ggplot")
                sns.set_theme()

                # Create a new figure
                fig = plt.figure(figsize=(15, 12))

                # Add subplots
                gs = fig.add_gridspec(2, 2)
                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[0, 1])
                ax3 = fig.add_subplot(gs[1, :])

                # Plot 1: Test accuracy vs ansatz depth
                for q in successful["n_qubits"].unique():
                    data = successful[successful["n_qubits"] == q]
                    sns.lineplot(
                        data=data,
                        x="ansatz_reps",
                        y="test_accuracy",
                        marker="o",
                        label=f"{q} qubits",
                        ax=ax1,
                    )

                ax1.set_xlabel("Ansatz Repetitions")
                ax1.set_ylabel("Test Accuracy (%)")
                ax1.set_title("Test Accuracy vs Ansatz Depth")
                ax1.grid(True)

                # Plot 2: Test accuracy vs feature map depth
                for q in successful["n_qubits"].unique():
                    data = successful[successful["n_qubits"] == q]
                    sns.lineplot(
                        data=data,
                        x="feature_map_reps",
                        y="test_accuracy",
                        marker="o",
                        label=f"{q} qubits",
                        ax=ax2,
                    )

                ax2.set_xlabel("Feature Map Repetitions")
                ax2.set_ylabel("Test Accuracy (%)")
                ax2.set_title("Test Accuracy vs Feature Map Depth")
                ax2.grid(True)

                # Plot 3: 3D scatter plot with test accuracy as color
                from mpl_toolkits.mplot3d import Axes3D

                ax3 = fig.add_subplot(gs[1, :], projection="3d")
                p = ax3.scatter(
                    successful["ansatz_reps"],
                    successful["feature_map_reps"],
                    successful["n_qubits"],
                    c=successful["test_accuracy"],
                    cmap="viridis",
                    s=100,
                )

                ax3.set_xlabel("Ansatz Repetitions")
                ax3.set_ylabel("Feature Map Repetitions")
                ax3.set_zlabel("Number of Qubits")
                ax3.set_title("Test Accuracy by Configuration")

                # Add colorbar
                cbar = fig.colorbar(p, ax=ax3, pad=0.1)
                cbar.set_label("Test Accuracy (%)")

                plt.tight_layout()

                # Save figure
                output_path = results_dir / "analysis_plot.png"
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                console.print(f"Analysis plot saved to: {output_path}")

                # Close figure to free memory
                plt.close(fig)

                # Create heatmap of test accuracy by ansatz and feature map repetitions
                plt.figure(figsize=(12, 10))

                # For each number of qubits, create a separate heatmap
                for i, q in enumerate(sorted(successful["n_qubits"].unique())):
                    qubit_data = successful[successful["n_qubits"] == q]

                    # Create pivot table
                    pivot = qubit_data.pivot_table(
                        index="ansatz_reps",
                        columns="feature_map_reps",
                        values="test_accuracy",
                    )

                    plt.subplot(len(successful["n_qubits"].unique()), 1, i + 1)
                    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis")
                    plt.title(f"Test Accuracy (%) for {q} Qubits")
                    plt.xlabel("Feature Map Repetitions")
                    plt.ylabel("Ansatz Repetitions")

                plt.tight_layout()
                heatmap_path = results_dir / "heatmap_plot.png"
                plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
                console.print(f"Heatmap plot saved to: {heatmap_path}")
                plt.close()

            except ImportError as e:
                console.print(
                    f"[yellow]Visualization libraries not available: {e}[/yellow]"
                )
                console.print("[yellow]Skipping visualization generation.[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Error generating visualizations: {e}[/yellow]")

        # Generate detailed report
        if output_file:
            with open(output_file, "w") as f:
                f.write("# Experiment Analysis Report\n\n")
                f.write(f"Analysis generated: {datetime.now().isoformat()}\n\n")
                f.write(f"Results directory: {results_dir}\n")
                f.write(f"Total experiments: {len(results_df)}\n")
                f.write(f"Successful experiments: {results_df['success'].sum()}\n\n")

                if results_df["success"].sum() > 0:
                    f.write("## Best Configuration\n\n")
                    f.write(f"- Qubits: {best_config['n_qubits']}\n")
                    f.write(f"- Ansatz repetitions: {best_config['ansatz_reps']}\n")
                    f.write(
                        f"- Feature map repetitions: {best_config['feature_map_reps']}\n"
                    )
                    f.write(f"- Test accuracy: {best_config['test_accuracy']:.2f}%\n")
                    f.write(
                        f"- Runtime: {best_config['runtime_seconds']:.1f} seconds\n\n"
                    )

                    f.write("## Performance by Configuration\n\n")
                    f.write(
                        "| Qubits | Ansatz | Feature Map | Test Acc (%) | Train Acc (%) | Val Acc (%) | Runtime (s) |\n"
                    )
                    f.write(
                        "|--------|--------|-------------|--------------|---------------|-------------|-------------|\n"
                    )

                    for _, row in successful.sort_values(
                        ["n_qubits", "ansatz_reps", "feature_map_reps"]
                    ).iterrows():
                        f.write(
                            f"| {row['n_qubits']} | {row['ansatz_reps']} | {row['feature_map_reps']} | "
                        )
                        f.write(f"{row['test_accuracy']:.2f} | ")
                        (
                            f.write(f"{row['final_train_accuracy']:.2f} | ")
                            if "final_train_accuracy" in row
                            and not pd.isna(row["final_train_accuracy"])
                            else f.write("N/A | ")
                        )
                        (
                            f.write(f"{row['final_val_accuracy']:.2f} | ")
                            if "final_val_accuracy" in row
                            and not pd.isna(row["final_val_accuracy"])
                            else f.write("N/A | ")
                        )
                        f.write(f"{row['runtime_seconds']:.1f} |\n")

                console.print(f"\nDetailed report saved to: {output_file}")

        return 0

    except Exception as e:
        console.print(f"[bold red]Error analyzing results: {e}[/bold red]")
        logger.exception("Error in analysis")
        return 1


def main():
    """Entry point for the experiment manager CLI."""
    return app()


if __name__ == "__main__":
    sys.exit(main())

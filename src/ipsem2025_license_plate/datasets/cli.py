"""Command-line interface for dataset management.

This module provides CLI commands for:
1. Downloading and managing datasets
2. Displaying dataset information
3. Previewing dataset samples
4. Validating custom datasets
"""

# pylint: disable=broad-exception-caught,unused-argument,too-many-locals

import os
import sys
import traceback
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from ..utils.logging_utils import configure_logging, get_logger
from .base import BaseDataset
from .custom import CustomImageDataset
from .emnist import EMNISTDataset

# Create Typer app
app = typer.Typer(
    help="IPSEM 2025 License Plate Dataset Management",
    add_completion=False,
)

# Create Rich console for pretty output
console = Console()
logger = get_logger(__name__)


def auto_detect_dataset_type(path: str) -> str:
    """Attempt to auto-detect the dataset type based on directory structure.

    Args:
        path: Path to the dataset

    Returns:
        Dataset type ("emnist", "custom", or "unknown")
    """
    path_obj = Path(path)

    # Check for EMNIST directory structure
    if (path_obj / "EMNIST").exists() or (path_obj / "emnist").exists():
        return "emnist"

    # Check for custom dataset structure (folders with images)
    potential_class_dirs = [d for d in path_obj.iterdir() if d.is_dir()]
    if potential_class_dirs:
        # Check if at least one directory contains image files
        for dir_path in potential_class_dirs:
            for ext in CustomImageDataset.SUPPORTED_EXTENSIONS:
                if list(dir_path.glob(f"*{ext}")):
                    return "custom"

    return "unknown"


def load_dataset(path: str, dataset_type: str = "auto") -> Optional[BaseDataset]:
    """Load a dataset based on its type."""
    try:
        if dataset_type == "auto":
            dataset_type = auto_detect_dataset_type(path)
            console.print(f"Auto-detected dataset type: [bold]{dataset_type}[/bold]")

        if dataset_type == "emnist":
            return EMNISTDataset(root=path, download=False, lazy_load=True)

        if dataset_type == "custom":
            return CustomImageDataset(root=path)

        console.print("[bold red]Unknown or undetected dataset type[/bold red]")
        return None

    except Exception as exc:
        console.print(f"[bold red]Error loading dataset:[/bold red] {exc}")
        logger.error("Failed to load dataset: %s", exc)
        return None


@app.command("download")
def download_dataset(
    dataset_name: str = typer.Option(
        "emnist",
        "--dataset-name",
        "-d",
        help="Dataset to download",
        show_choices=True,
        case_sensitive=False,
    ),
    output_dir: str = typer.Option(
        "data", "--output-dir", "-o", help="Directory to store the dataset"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-download even if dataset exists"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """Download a dataset for license plate character recognition.

    Currently supports EMNIST dataset (36 classes: 0-9, A-Z).
    """
    # Configure logging - use log_level instead of verbose
    log_level = "DEBUG" if verbose else "INFO"
    configure_logging(level=log_level)

    console.print(
        f"Downloading [bold]{dataset_name}[/bold] dataset to [bold]{output_dir}[/bold]"
    )

    try:
        if dataset_name.lower() == "emnist":
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Downloading EMNIST dataset..."),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
                transient=False,  # This ensures progress stays on one line
            ) as progress:
                task = progress.add_task("Downloading", total=None)

                # Download EMNIST dataset
                dataset = EMNISTDataset(
                    root=output_dir,
                    download=True,
                )

                progress.update(task, completed=100, total=100)

            console.print(
                f"\n[bold green]Successfully downloaded EMNIST dataset[/bold green] with "
                f"{len(dataset)} samples (36 classes: 0-9, A-Z)"
            )

            # Show some additional information in a table
            class_mapping = dataset.get_class_mapping()

            table = Table(title="EMNIST Class Mapping")
            table.add_column("Index", justify="right", style="cyan")
            table.add_column("Character", justify="center", style="green")

            for idx in range(min(10, len(class_mapping))):  # Show first 10 classes
                table.add_row(str(idx), class_mapping[idx])

            table.add_row("...", "...")

            for idx in range(10, 36, 5):  # Show a few letter classes
                table.add_row(str(idx), class_mapping[idx])

            console.print(table)

            console.print(
                Panel(
                    "Dataset is ready for use. Try:\n"
                    f"[bold]ipsem2025-dataset info --dataset-path {output_dir}[/bold]\n"
                    f"[bold]ipsem2025-dataset preview --dataset-path {output_dir}[/bold]",
                    title="Next Steps",
                    expand=False,
                )
            )

            return 0

        if dataset_name.lower() == "custom":
            console.print(
                "[bold yellow]Custom dataset download is not supported.[/bold yellow] "
                "Please use a local directory with class folders."
            )
            return 1

        console.print(f"[bold red]Unknown dataset type:[/bold red] {dataset_name}")
        return 1

    except Exception as exc:
        console.print(f"[bold red]Error downloading dataset:[/bold red] {exc}")
        logger.error("Error downloading dataset: %s", exc)
        if verbose:
            console.print(traceback.format_exc())
        return 1


@app.command("info")
def info_command(
    dataset_path: str = typer.Option(
        ..., "--dataset-path", "-p", help="Path to the dataset"
    ),
    dataset_type: str = typer.Option(
        "auto",
        "--dataset-type",
        "-t",
        help="Type of dataset",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """Display information about a dataset.

    Shows details such as:
    - Number of samples and classes
    - Class mapping and distribution
    - Image dimensions
    """
    # Configure logging - use log_level instead of verbose
    log_level = "DEBUG" if verbose else "INFO"
    configure_logging(level=log_level)

    dataset = load_dataset(dataset_path, dataset_type)
    if dataset is None:
        return 1

    # Get basic information
    num_samples = len(dataset)
    num_classes = dataset.get_num_classes()
    image_dims = dataset.get_image_dimensions()
    class_mapping = dataset.get_class_mapping()

    # Display basic information
    console.print(
        Panel(
            f"[bold]Type:[/bold] {dataset.__class__.__name__}\n"
            f"[bold]Samples:[/bold] {num_samples}\n"
            f"[bold]Classes:[/bold] {num_classes}\n"
            f"[bold]Image Dimensions:[/bold] {image_dims}",
            title="Dataset Information",
            expand=False,
        )
    )

    # Display class mapping in a table
    table = Table(title="Class Mapping")
    table.add_column("Index", justify="right", style="cyan")
    table.add_column("Name", justify="center", style="green")

    for idx, name in sorted(class_mapping.items()):
        if idx < 10 or idx % 5 == 0:  # Show a subset for large datasets
            table.add_row(str(idx), name)

    console.print(table)

    # Calculate class distribution
    class_counts = calculate_class_distribution(dataset, num_samples)

    # Display class distribution in a table
    table = Table(title="Class Distribution")
    table.add_column("Class", justify="left", style="cyan")
    table.add_column("Samples", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="yellow")

    for label, count in sorted(class_counts.items()):
        percentage = 100 * count / num_samples
        table.add_row(
            class_mapping.get(label, str(label)), str(count), f"{percentage:.1f}%"
        )

    console.print(table)

    # Suggest next steps
    console.print(
        Panel(
            "Try previewing samples from this dataset:\n"
            f"[bold]ipsem2025-dataset preview --dataset-path {dataset_path} "
            f"--num-samples 8[/bold]",
            title="Next Steps",
            expand=False,
        )
    )

    return 0


def calculate_class_distribution(dataset, num_samples):
    """Calculate the distribution of classes in the dataset.

    Args:
        dataset: The dataset to analyze
        num_samples: Number of samples in the dataset

    Returns:
        Dictionary mapping class labels to counts
    """
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Calculating class distribution...", total=num_samples)

        class_counts = {}
        for i in range(num_samples):
            _, label = dataset[i]
            class_counts[label] = class_counts.get(label, 0) + 1
            progress.update(task, advance=1)

    return class_counts


@app.command("preview")
def preview_command(
    dataset_path: str = typer.Option(
        ..., "--dataset-path", "-p", help="Path to the dataset"
    ),
    dataset_type: str = typer.Option(
        "auto", "--dataset-type", "-t", help="Type of dataset"
    ),
    num_samples: int = typer.Option(
        8, "--num-samples", "-n", help="Number of samples to display"
    ),
    save_plot: Optional[str] = typer.Option(
        None, "--save-plot", "-s", help="Save preview to file instead of displaying"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """Preview samples from a dataset.

    Displays sample images with their labels or saves the preview to a file.
    """
    # Configure logging - use log_level instead of verbose
    log_level = "DEBUG" if verbose else "INFO"
    configure_logging(level=log_level)

    console.print(f"Loading dataset from [bold]{dataset_path}[/bold]")

    dataset = load_dataset(dataset_path, dataset_type)
    if dataset is None:
        return 1

    # Preview dataset
    console.print(f"Generating preview with [bold]{num_samples}[/bold] samples")

    # Limit number of samples
    num_samples = min(num_samples, len(dataset))

    # Set up the plot
    fig, axes = plt.subplots(
        2, num_samples // 2, figsize=(12, 5), constrained_layout=True
    )
    axes = axes.flatten()

    # Get class mapping
    class_mapping = dataset.get_class_mapping()

    # Display samples
    for i in range(num_samples):
        # Get a sample
        img, label = dataset[i]

        # Convert tensor to numpy array
        if isinstance(img, torch.Tensor):
            if img.shape[0] == 1:  # Grayscale
                img_np = img.squeeze().numpy()
                cmap = "gray"
            else:  # RGB
                img_np = img.permute(1, 2, 0).numpy()
                cmap = None
        else:
            img_np = img
            cmap = None

        # Display image
        axes[i].imshow(img_np, cmap=cmap)
        axes[i].set_title(f"{class_mapping.get(label, str(label))}")
        axes[i].axis("off")

    # Set title for the figure
    fig.suptitle(
        f"Dataset Preview: {dataset.__class__.__name__} ({len(dataset)} samples, "
        f"{dataset.get_num_classes()} classes)",
        fontsize=14,
    )

    # Save or display
    if save_plot:
        plt.savefig(save_plot, dpi=300, bbox_inches="tight")
        console.print(f"[bold green]Preview saved to[/bold green] {save_plot}")
    else:
        plt.show()

    return 0


@app.command("validate")
def validate_command(
    dataset_path: str = typer.Option(
        ..., "--dataset-path", "-p", help="Path to the dataset to validate"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """Validate a custom dataset structure.

    Checks if the dataset follows the expected structure and reports any issues.
    """
    # Configure logging - use log_level instead of verbose
    log_level = "DEBUG" if verbose else "INFO"
    configure_logging(level=log_level)

    console.print(f"Validating custom dataset at [bold]{dataset_path}[/bold]")

    path_obj = Path(dataset_path)
    if not path_obj.exists():
        console.print(
            f"[bold red]Dataset path does not exist:[/bold red] {dataset_path}"
        )
        return 1

    if not path_obj.is_dir():
        console.print(
            f"[bold red]Dataset path is not a directory:[/bold red] {dataset_path}"
        )
        return 1

    # Check for class directories
    class_dirs = [d for d in path_obj.iterdir() if d.is_dir()]
    if not class_dirs:
        console.print(
            f"[bold red]No class directories found in[/bold red] {dataset_path}"
        )
        return 1

    console.print(f"Found [bold]{len(class_dirs)}[/bold] potential class directories")

    # Check each class directory for valid images
    valid_extensions = CustomImageDataset.SUPPORTED_EXTENSIONS
    issues_found = False

    # Create a table for results
    table = Table(title="Class Directory Validation Results")
    table.add_column("Class", style="cyan")
    table.add_column("Valid Images", justify="right", style="green")
    table.add_column("Invalid Files", justify="right", style="yellow")
    table.add_column("Status", style="bold")

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Validating class directories...", total=len(class_dirs)
        )

        for class_dir in class_dirs:
            # Check for valid images
            all_files = list(class_dir.iterdir())
            image_files = []
            invalid_files = []

            for file in all_files:
                if file.suffix.lower() in valid_extensions:
                    image_files.append(file)
                elif file.is_file():
                    invalid_files.append(file)

            # Determine status
            if not image_files:
                status = "[bold red]ERROR: No valid images[/bold red]"
                issues_found = True
            elif invalid_files:
                status = "[bold yellow]WARNING: Has invalid files[/bold yellow]"
            else:
                status = "[bold green]OK[/bold green]"

            # Add to table
            table.add_row(
                class_dir.name, str(len(image_files)), str(len(invalid_files)), status
            )

            progress.update(task, advance=1)

    console.print(table)

    if issues_found:
        console.print("[bold red]Issues were found in the dataset structure[/bold red]")
        console.print(
            "Please ensure each class directory contains valid image files. "
            f"Supported formats: {', '.join(valid_extensions)}"
        )
        return 1

    console.print(
        "[bold green]Dataset validation successful: no issues found[/bold green]"
    )

    # Suggest next steps
    console.print(
        Panel(
            "Your custom dataset looks good! Try previewing it:\n"
            f"[bold]ipsem2025-dataset preview --dataset-path {dataset_path}[/bold]",
            title="Next Steps",
            expand=False,
        )
    )

    return 0


def main():
    """Main entry point for the dataset CLI."""
    try:
        return app()
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        logger.error("Unhandled exception in CLI: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

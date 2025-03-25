"""
CLI interface for the IPSEM 2025 License Plate recognition project.

This module provides a comprehensive command-line interface for all
functionality related to the license plate recognition system.
"""

import argparse
import logging
import sys
from typing import List, Optional

from ipsem2025_license_plate.qnet.dataset_utils import load_and_prepare_dataset
from ipsem2025_license_plate.qnet.hybrid_net import train_and_evaluate
from ipsem2025_license_plate.utils.cli_utils import (
    add_common_logging_args,
    add_common_training_args,
    get_logging_config_from_args,
)
from ipsem2025_license_plate.utils.logging_utils import configure_logging, get_logger

# Get module logger
logger = get_logger(__name__)


def train_command(args: argparse.Namespace) -> None:
    """Handle the train subcommand."""
    logger.info(
        "Starting training with batch size %s, %s epochs", args.batch_size, args.epochs
    )

    # Configure logging for this command
    logging_config = get_logging_config_from_args(args)

    _, accuracy = train_and_evaluate(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        log_level=logging_config["level"],
    )
    logger.info("Training completed with %.2f%% accuracy", accuracy)


def dataset_command(args: argparse.Namespace) -> None:
    """Handle the dataset subcommand."""
    if args.info:
        logger.info("Getting dataset information...")
        train_loader, test_loader, num_classes = load_and_prepare_dataset(
            batch_size=args.batch_size
        )
        # Get dataset sizes
        train_size = len(train_loader.dataset)
        test_size = len(test_loader.dataset)

        logger.info("Dataset loaded successfully:")
        logger.info("  - Number of classes: %s", num_classes)
        logger.info("  - Training samples: %s", train_size)
        logger.info("  - Testing samples: %s", test_size)
        logger.info("  - Batch size: %s", args.batch_size)
        logger.info("  - Batches per epoch: %s", len(train_loader))

        # Show a batch sample if requested
        if args.show_sample:
            images, labels = next(iter(train_loader))
            logger.info("Sample batch shape: %s", images.shape)
            sample_labels = labels[: min(8, len(labels))].tolist()
            logger.info("Sample labels: %s", sample_labels)


def add_train_parser(subparsers) -> None:
    """Add parser for the train subcommand."""
    parser = subparsers.add_parser(
        "train", help="Train the hybrid quantum-classical model"
    )

    # Add shared training arguments
    add_common_training_args(parser)
    add_common_logging_args(parser)

    parser.set_defaults(func=train_command)


def add_dataset_parser(subparsers) -> None:
    """Add parser for the dataset subcommand."""
    parser = subparsers.add_parser("dataset", help="Dataset-related operations")

    parser.add_argument(
        "--info", action="store_true", help="Show information about the dataset"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for dataset operations"
    )
    parser.add_argument(
        "--show-sample",
        action="store_true",
        help="Show a sample batch from the dataset",
    )

    # Add shared logging arguments
    add_common_logging_args(parser)

    parser.set_defaults(func=dataset_command)


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="IPSEM 2025 License Plate Recognition with Quantum Neural Networks"
    )

    # Add global logging options
    add_common_logging_args(parser)

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        title="subcommands",
        description="valid subcommands",
        help="additional help",
        dest="subcommand",
    )

    # Add parsers for each subcommand
    add_train_parser(subparsers)
    add_dataset_parser(subparsers)

    # Parse arguments and run the appropriate command
    args = parser.parse_args(argv)

    # Configure logging using shared utility
    logging_config = get_logging_config_from_args(args)
    configure_logging(**logging_config)

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    try:
        args.func(args)
        return 0
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("An error occurred: %s", e)
        if args.verbose or logging_config["level"] <= logging.DEBUG:
            logger.exception("Detailed traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())

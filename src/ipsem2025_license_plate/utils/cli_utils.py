"""
CLI and argument parsing utilities shared across the project.

This module provides common argument parsing functions used by
both the main module and CLI interfaces.
"""

import argparse
import logging
from typing import Any, Dict, Optional


def add_common_training_args(parser: argparse.ArgumentParser) -> None:
    """
    Add common training-related arguments to a parser.

    Args:
        parser: The argument parser to add arguments to
    """
    # Training parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="number of epochs to train (default: 2)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 0.001)"
    )

    # Device selection
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )


def add_common_logging_args(parser: argparse.ArgumentParser) -> None:
    """
    Add logging-related arguments to a parser.

    Args:
        parser: The argument parser to add arguments to
    """
    log_group = parser.add_argument_group("Logging options")
    log_group.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level",
    )
    log_group.add_argument(
        "--no-console-log", action="store_true", help="Disable logging to console"
    )
    log_group.add_argument(
        "--log-file", type=str, default=None, help="Log to the specified file"
    )
    log_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output (same as --log-level=DEBUG)",
    )


def get_logging_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Get logging configuration from command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Dictionary of logging configuration parameters
    """
    log_level = getattr(logging, args.log_level) if not args.verbose else logging.DEBUG
    log_to_console = not args.no_console_log
    log_to_file = args.log_file is not None

    return {
        "level": log_level,
        "log_to_console": log_to_console,
        "log_to_file": log_to_file,
        "log_file": args.log_file,
    }


def create_training_args_parser(
    description: Optional[str] = None,
) -> argparse.ArgumentParser:
    """
    Create an argument parser with common training and logging arguments.

    Args:
        description: Optional description for the parser

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(description=description)
    add_common_training_args(parser)
    add_common_logging_args(parser)
    return parser

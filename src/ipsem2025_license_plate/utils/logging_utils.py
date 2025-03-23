"""
Logging utilities for the IPSEM 2025 License Plate recognition project.

This module provides configurable logging functionality allowing users to
direct log output to stdout, a file, both, or neither.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional, Union

# Default log formatter
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_LEVEL = logging.INFO


# pylint: disable=too-many-arguments,too-many-positional-arguments
def configure_logging(
    level: int = DEFAULT_LOG_LEVEL,
    log_to_console: bool = True,
    log_to_file: bool = False,
    log_file: Optional[Union[str, Path]] = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    loggers: Optional[List[str]] = None,
) -> None:
    """
    Configure the logging system for the package.

    Args:
        level: The logging level (e.g., logging.INFO, logging.DEBUG)
        log_to_console: Whether to log to console (stdout)
        log_to_file: Whether to log to a file
        log_file: The log file path (if log_to_file is True). If None, a default path
                  in the current directory will be used.
        log_format: The log message format
        date_format: The format for timestamps
        loggers: List of specific logger names to configure. If None, configures the root logger.
    """
    handlers = []
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    # Create console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    # Create file handler if requested
    if log_to_file:
        if log_file is None:
            # Default log file in the current directory
            log_file = Path("ipsem2025_license_plate.log")
        else:
            log_file = Path(log_file)

        # Create directory for log file if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        # Type annotation fix for mypy
        handlers.append(file_handler)  # type: ignore

    # Configure the specified loggers or the root logger
    if loggers:
        for logger_name in loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)

            # Remove any existing handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

            # Add new handlers
            for handler in handlers:
                logger.addHandler(handler)

            # Don't propagate to avoid duplicate logging
            logger.propagate = False
    else:
        # Configure the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Remove any existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add new handlers
        for handler in handlers:
            root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Name of the logger to get

    Returns:
        A logger instance
    """
    return logging.getLogger(name)

"""Main module for IPSEM 2025 License Plate recognition.

This module provides the main entry point for the license plate
recognition system using a hybrid quantum-classical neural network.
"""

import logging
import sys

import torch

from .qnet.hybrid_net import train_and_evaluate
from .utils.cli_utils import create_training_args_parser, get_logging_config_from_args
from .utils.logging_utils import configure_logging, get_logger

# Get the module logger
logger = get_logger(__name__)


def main():
    """Run the license plate recognition system with specified parameters."""
    # Create parser with shared arguments
    parser = create_training_args_parser(
        "IPSEM 2025 License Plate Recognition with Quantum Neural Networks"
    )
    args = parser.parse_args()

    # Configure logging using shared utility
    logging_config = get_logging_config_from_args(args)
    configure_logging(**logging_config)

    # Check CUDA availability
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        logger.info("Using CUDA for training")
    else:
        logger.info("Using CPU for training")

    try:
        # Run the training and evaluation
        model, accuracy = train_and_evaluate(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            log_level=logging_config["level"],
        )

        logger.info("Training completed with %.2f%% accuracy", accuracy)
        return model
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("An error occurred: %s", e)
        if args.verbose or logging_config["level"] <= logging.DEBUG:
            logger.exception("Detailed traceback:")
        return None


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)

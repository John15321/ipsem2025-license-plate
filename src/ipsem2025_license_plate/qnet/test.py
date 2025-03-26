"""Testing and evaluation functionality for hybrid models."""

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def evaluate_model(
    model: nn.Module, test_loader: DataLoader, device: str
) -> Dict[str, float]:
    """Evaluate model on test dataset."""
    logger.info("Starting model evaluation")
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    class_correct = torch.zeros(model.num_classes)
    class_total = torch.zeros(model.num_classes)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            total_loss += loss.item() * images.size(0)

            # Per-class accuracy
            for label, pred in zip(labels, predicted):
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1

    # Calculate metrics
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    # Calculate per-class accuracies
    class_accuracies = {}
    for i in range(model.num_classes):
        if class_total[i] > 0:
            class_accuracies[f"class_{i}_accuracy"] = (
                100.0 * class_correct[i].item() / class_total[i].item()
            )

    metrics = {"test_loss": avg_loss, "test_accuracy": accuracy, **class_accuracies}

    logger.info(
        f"Evaluation complete - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
    )
    return metrics

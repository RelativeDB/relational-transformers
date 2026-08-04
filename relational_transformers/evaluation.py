"""Evaluators for prediction and regression."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .training import RelationalExample


def _labels(examples: Sequence[RelationalExample]) -> np.ndarray:
    if not examples:
        raise ValueError("evaluation requires at least one example")
    return np.asarray([example.label for example in examples])


def _inputs(model, examples: Sequence[RelationalExample]) -> list:
    return [model._batch(example.input, target=example.target) for example in examples]


@dataclass
class BinaryClassificationEvaluator:
    """Evaluate binary predictions at a configurable probability threshold."""

    examples: Sequence[RelationalExample]
    threshold: float = 0.5
    task_head: str | None = None

    def __call__(self, model) -> dict[str, float]:
        labels = _labels(self.examples).astype(bool)
        probabilities = np.asarray(
            model.predict(
                _inputs(model, self.examples),
                task_head=self.task_head,
            )
        ).reshape(-1)
        predicted = probabilities >= self.threshold
        true_positive = int((predicted & labels).sum())
        false_positive = int((predicted & ~labels).sum())
        false_negative = int((~predicted & labels).sum())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, np.finfo(float).eps)
        return {
            "accuracy": float((predicted == labels).mean()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


@dataclass
class RegressionEvaluator:
    """Compute MAE, RMSE, and R² for regression or forecasting predictions."""

    examples: Sequence[RelationalExample]
    task_head: str | None = None

    def __call__(self, model) -> dict[str, float]:
        labels = _labels(self.examples).astype(np.float64).reshape(-1)
        predictions = np.asarray(
            model.predict(
                _inputs(model, self.examples),
                task_head=self.task_head,
                activation="identity",
            ),
            dtype=np.float64,
        ).reshape(-1)
        residual = predictions - labels
        total = np.square(labels - labels.mean()).sum()
        r2 = 1.0 - np.square(residual).sum() / total if total else math.nan
        return {
            "mae": float(np.abs(residual).mean()),
            "rmse": float(np.sqrt(np.square(residual).mean())),
            "r2": float(r2),
        }


class SequentialEvaluator:
    """Run multiple evaluators and merge their non-overlapping metrics."""

    def __init__(self, evaluators: Sequence) -> None:
        self.evaluators = list(evaluators)
        if not self.evaluators:
            raise ValueError("SequentialEvaluator requires at least one evaluator")

    def __call__(self, model) -> dict[str, float]:
        metrics = {}
        for evaluator in self.evaluators:
            values = evaluator(model)
            overlap = metrics.keys() & values.keys()
            if overlap:
                raise ValueError(f"duplicate evaluator metrics: {', '.join(sorted(overlap))}")
            metrics.update(values)
        return metrics

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from relational_transformers import (
    AblationEvaluator,
    BinaryClassificationEvaluator,
    BinaryClassificationLoss,
    MulticlassClassificationLoss,
    MultilabelClassificationLoss,
    RegressionEvaluator,
    RegressionLoss,
    RelationalDataset,
    RelationalExample,
    RelationalTransformer,
    SequentialEvaluator,
)


def examples(customer_context_factory):
    return [
        RelationalExample(customer_context_factory(customer_id=44_904_001), 1),
        RelationalExample(
            customer_context_factory(
                customer_id=44_904_101,
                support_summary="Asked about downgrading next renewal",
            ),
            0,
        ),
    ]


def test_relational_dataset_builds_from_inputs_and_labels(customer_context_factory):
    inputs = [
        customer_context_factory(customer_id=44_905_001),
        customer_context_factory(customer_id=44_905_101),
    ]
    dataset = RelationalDataset.from_inputs(inputs, [1, 0])

    assert len(dataset) == 2
    assert list(dataset)[0].input is inputs[0]
    assert dataset[1].label == 0
    with pytest.raises(ValueError, match="same length"):
        RelationalDataset.from_inputs(inputs, [1])
    with pytest.raises(ValueError, match="at least one"):
        RelationalDataset([])
    with pytest.raises(TypeError, match="RelationalExample"):
        RelationalDataset([object()])


def test_public_losses_cover_supported_task_shapes():
    binary = BinaryClassificationLoss()(torch.tensor([0.0]), torch.tensor([1.0]))
    multiclass = MulticlassClassificationLoss()(torch.tensor([[2.0, 0.5, -1.0]]), torch.tensor([0]))
    multilabel = MultilabelClassificationLoss()(
        torch.tensor([[1.0, -1.0]]), torch.tensor([[1.0, 0.0]])
    )
    regression = RegressionLoss(delta=0.5)(torch.tensor([2.0]), torch.tensor([1.5]))

    for loss in (binary, multiclass, multilabel, regression):
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss >= 0


def test_prediction_and_ablation_evaluators_return_finite_metrics(
    tiny_checkpoint, customer_context_factory
):
    model = RelationalTransformer(tiny_checkpoint, device="cpu")
    evaluation_examples = examples(customer_context_factory)
    binary = BinaryClassificationEvaluator(evaluation_examples)
    regression = RegressionEvaluator(evaluation_examples)
    ablation = AblationEvaluator(evaluation_examples, {"support": [11, 12]})

    binary_metrics = binary(model)
    regression_metrics = regression(model)
    ablation_metrics = ablation(model)

    assert set(binary_metrics) == {"accuracy", "precision", "recall", "f1"}
    assert 0.0 <= binary_metrics["accuracy"] <= 1.0
    assert regression_metrics["mae"] >= 0.0
    assert regression_metrics["rmse"] >= 0.0
    assert math.isfinite(regression_metrics["r2"])
    assert ablation_metrics["support_mean_absolute_delta"] >= 0.0


def test_sequential_evaluator_merges_metrics_and_rejects_collisions(
    tiny_checkpoint, customer_context_factory
):
    model = RelationalTransformer(tiny_checkpoint, device="cpu")
    evaluation_examples = examples(customer_context_factory)
    combined = SequentialEvaluator(
        [
            BinaryClassificationEvaluator(evaluation_examples),
            AblationEvaluator(evaluation_examples, {"support": [11, 12]}),
        ]
    )

    metrics = combined(model)

    assert "accuracy" in metrics
    assert "support_mean_delta" in metrics
    with pytest.raises(ValueError, match="at least one evaluator"):
        SequentialEvaluator([])
    duplicate = SequentialEvaluator([BinaryClassificationEvaluator(evaluation_examples)] * 2)
    with pytest.raises(ValueError, match="duplicate evaluator metrics"):
        duplicate(model)


def test_regression_evaluator_reports_nan_r2_for_constant_labels(
    tiny_checkpoint, customer_context_factory
):
    model = RelationalTransformer(tiny_checkpoint, task="regression", device="cpu")
    constant = [
        RelationalExample(customer_context_factory(customer_id=44_906_001), 3.0),
        RelationalExample(customer_context_factory(customer_id=44_906_101), 3.0),
    ]

    assert np.isnan(RegressionEvaluator(constant)(model)["r2"])

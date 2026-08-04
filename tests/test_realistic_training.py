from __future__ import annotations

import numpy as np
import pytest
import torch

from relational_transformers import (
    RelationalExample,
    RelationalTrainer,
    RelationalTrainingArguments,
    RelationalTransformer,
    TaskHead,
)


def customer_examples(customer_context_factory):
    return [
        RelationalExample(
            customer_context_factory(
                customer_id=44_903_200 + index * 20,
                tenure_months=tenure,
                plan=plan,
                order_amounts=orders,
                support_summary=support,
            ),
            label,
        )
        for index, (tenure, plan, orders, support, label) in enumerate(
            [
                (2.0, "starter monthly", (9.0,), "Cannot cancel trial", 1),
                (38.0, "business annual", (180.0, 95.0), None, 0),
                (4.0, "starter monthly", (12.0,), "Card declined three times", 1),
                (51.0, "enterprise annual", (900.0, 750.0), None, 0),
                (7.0, "business monthly", (45.0,), "Service unavailable", 1),
                (29.0, "business annual", (220.0, 130.0), None, 0),
            ]
        )
    ]


def test_head_tuning_keeps_backbone_frozen_and_round_trips(
    tiny_checkpoint, customer_context_factory, tmp_path
):
    model = RelationalTransformer(tiny_checkpoint, device="cpu")
    examples = customer_examples(customer_context_factory)
    before = {name: value.clone() for name, value in model.model.state_dict().items()}

    head = model.fit_head(
        examples,
        task="churn_risk",
        problem_type="binary",
        epochs=40,
        learning_rate=0.05,
        weight_decay=0.0,
    )

    assert all(torch.equal(before[name], value) for name, value in model.model.state_dict().items())
    predictions = model.predict([example.input for example in examples], task_head="churn_risk")
    assert predictions.shape == (6, 1)
    assert torch.isfinite(torch.from_numpy(predictions)).all()

    destination = tmp_path / "churn-head"
    head.save_pretrained(destination)
    restored = TaskHead.from_pretrained(destination)
    features = model.encode(
        [example.input for example in examples],
        output_value="target_features",
        convert_to_numpy=False,
    )
    with torch.inference_mode():
        torch.testing.assert_close(restored(features), head(features))


def test_multiclass_head_returns_one_distribution_per_customer(
    tiny_checkpoint, customer_context_factory
):
    model = RelationalTransformer(tiny_checkpoint, device="cpu")
    examples = customer_examples(customer_context_factory)
    for index, example in enumerate(examples):
        example.label = index % 3

    model.fit_head(
        examples,
        task="retention_action",
        num_labels=3,
        problem_type="multiclass",
        epochs=5,
        learning_rate=0.02,
    )
    probabilities = model.predict(
        [example.input for example in examples[:2]], task_head="retention_action"
    )

    assert probabilities.shape == (2, 3)
    torch.testing.assert_close(
        torch.from_numpy(probabilities.sum(axis=1)),
        torch.ones(2),
        atol=1e-6,
        rtol=1e-6,
    )


def test_full_finetuning_updates_backbone_and_produces_reloadable_checkpoint(
    tiny_checkpoint, customer_context_factory, tmp_path
):
    model = RelationalTransformer(tiny_checkpoint, device="cpu")
    examples = customer_examples(customer_context_factory)[:4]
    before = {name: value.clone() for name, value in model.model.state_dict().items()}
    output_dir = tmp_path / "fine-tuned-churn"
    trainer = RelationalTrainer(
        model=model,
        args=RelationalTrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=1e-3,
            save_strategy="epoch",
        ),
        train_dataset=examples,
        problem_type="binary",
    )

    metrics = trainer.train()

    assert metrics["steps"] == 8
    assert metrics["train_loss"] >= 0.0
    assert any(
        not torch.equal(before[name], value) for name, value in model.model.state_dict().items()
    )
    restored = RelationalTransformer(output_dir, device="cpu")
    expected = model.predict(examples[0].input, activation="identity")
    actual = restored.predict(examples[0].input, activation="identity")
    assert actual == pytest.approx(expected, abs=1e-7)


def test_training_rejects_zero_gradient_accumulation(
    tiny_checkpoint, customer_context_factory, tmp_path
):
    model = RelationalTransformer(tiny_checkpoint, device="cpu")
    trainer = RelationalTrainer(
        model=model,
        args=RelationalTrainingArguments(
            output_dir=str(tmp_path),
            gradient_accumulation_steps=0,
        ),
        train_dataset=customer_examples(customer_context_factory)[:1],
    )

    with pytest.raises(ValueError, match="at least 1"):
        trainer.train()


def test_head_and_full_training_accept_raw_cell_vectors_with_explicit_targets(
    tiny_checkpoint, tmp_path
):
    rng = np.random.default_rng(23)
    examples = [
        RelationalExample(rng.normal(size=(cells, 8)).astype(np.float32), label, target=0)
        for cells, label in ((3, 0), (5, 1))
    ]
    head_model = RelationalTransformer(tiny_checkpoint, device="cpu")
    head_model.fit_head(examples, task="raw", epochs=1)
    predictions = [
        head_model.predict(example.input, target=example.target, task_head="raw")
        for example in examples
    ]
    assert all(np.isfinite(prediction).all() for prediction in predictions)

    full_model = RelationalTransformer(tiny_checkpoint, device="cpu")
    trainer = RelationalTrainer(
        model=full_model,
        args=RelationalTrainingArguments(
            output_dir=str(tmp_path / "raw-finetune"),
            num_train_epochs=1,
            per_device_train_batch_size=2,
        ),
        train_dataset=examples,
    )
    assert trainer.train()["steps"] == 1

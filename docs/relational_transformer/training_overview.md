# Training Overview

Relational Transformers supports frozen-backbone head tuning, complete RT-J
fine-tuning, and ordinary PyTorch loops.

## Why Fine-tune?

Fine-tuning adapts the model to a new cell encoder, schema distribution, target
definition, or domain. Start with a frozen task head when the published
backbone already produces useful target features; update the full model only
when the relational representation itself must change.

## Training Components

Training combines a model, a `RelationalDataset` or sequence of
`RelationalExample` objects, a problem-appropriate loss, training arguments,
and optional evaluators.

## Model

Load a trainable PyTorch model with
`RelationalTransformer("RelativeDB/rt-j-fp16", backend="torch")`. Quantized
checkpoints can run portable inference, but full fine-tuning should begin from
FP16 weights.

## Dataset

Each example contains a complete context and label. Contexts may have different
lengths but must share the same embedding width and checkpoint contract.

### Dataset Format

Use a `RelationalBatch` to preserve typed number, text, datetime, and boolean
channels plus node, column, table, and foreign-key relations.

## Loss Function

Choose `binary`, `multiclass`, `multilabel`, `regression`, or `forecasting`.
See the [Loss Overview](loss_overview.md) for shapes and defaults.

## Training Arguments

`RelationalTrainingArguments` configures epochs, per-device batch size,
learning rate, weight decay, gradient accumulation, clipping, seeding, and
checkpoint output.

## Evaluator

Use `BinaryClassificationEvaluator`, `RegressionEvaluator`, or
`AblationEvaluator` before and after training. `SequentialEvaluator` combines
non-overlapping metric dictionaries.

## Trainer

`RelationalTrainer` fine-tunes every RT-J parameter and writes a regular
Hugging Face-style safetensors checkpoint. `model.fit_head()` encodes examples
once and updates only a named task head.

## End-to-End Example

```python
from relational_transformers import (
    BinaryClassificationEvaluator,
    RelationalTrainer,
    RelationalTrainingArguments,
)

before = BinaryClassificationEvaluator(validation_examples)(model)
trainer = RelationalTrainer(
    model=model,
    args=RelationalTrainingArguments(
        output_dir="models/churn",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
    ),
    train_dataset=train_examples,
    problem_type="binary",
)
trainer.train()
after = BinaryClassificationEvaluator(validation_examples)(model)
```

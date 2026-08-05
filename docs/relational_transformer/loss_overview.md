# Loss Overview

The built-in trainer selects a task loss from `problem_type`. The same loss modules are
public for ordinary PyTorch loops, and `loss_for(problem_type)` returns the standard
module for a supported type.

## Loss Table

| Problem type | Loss | Output shape | Label shape |
| --- | --- | --- | --- |
| `binary` | `BinaryClassificationLoss` | `[batch]` or `[batch, 1]` | `[batch]` |
| `multiclass` | `MulticlassClassificationLoss` | `[batch, classes]` | `[batch]` integer IDs |
| `multilabel` | `MultilabelClassificationLoss` | `[batch, labels]` | `[batch, labels]` |
| `regression` | `RegressionLoss` | `[batch]` | `[batch]` |
| `forecasting` | `RegressionLoss` | `[batch]` | `[batch]` |
| ranking (grouped) | `ListwiseRankingLoss` | `[candidates]` + group offsets | `[candidates]` relevance |

An unsupported problem type raises `ValueError` from `loss_for`.

## Classification

Binary and multilabel tasks use binary cross entropy with logits. Multiclass tasks use
cross entropy over mutually exclusive class logits. Pass raw logits to loss modules; the
sigmoid or softmax lives inside the loss, and activating twice silently flattens
gradients.

Binary labels arrive as anything castable to float. Multiclass labels are integer class
IDs, and multilabel labels are multi-hot float rows matching the logit shape.

## Regression and Forecasting

`RegressionLoss` uses Huber loss with a configurable `delta`, which bounds the influence
of outlier targets compared to plain mean squared error. Normalize targets during data
preparation when their scale is large or highly skewed, and store the inverse transform
with the trained artifact so predictions can be mapped back to real units.

```python
from relational_transformers import RegressionLoss

loss = RegressionLoss(delta=2.0)
value = loss(predictions, labels)
```

## Custom Loss Functions

A custom loss is any `torch.nn.Module` accepting predictions and labels. Two model
outputs are the natural attachment points:

- `target_scores` when adapting the published scalar decoder, for example with a class
  weight or a ranking objective;
- `target_features` when the task needs its own head, giving the loss a `[batch, 512]`
  feature to build on.

```python
logits = model.forward(batch, output="target_scores").scores
loss = my_loss(logits, labels)
loss.backward()
```

The [custom PyTorch loop](training/examples.md#custom-pytorch-loop) example shows where
this slots into a full training step.

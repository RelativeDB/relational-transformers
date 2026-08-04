# Loss Overview

The built-in trainer selects a task loss from `problem_type`. The same loss
modules are public for ordinary PyTorch loops.

## Loss Table

| Problem type | Loss | Output shape | Label shape |
| --- | --- | --- | --- |
| `binary` | `BinaryClassificationLoss` | `[batch]` or `[batch, 1]` | `[batch]` |
| `multiclass` | `MulticlassClassificationLoss` | `[batch, classes]` | `[batch]` integer IDs |
| `multilabel` | `MultilabelClassificationLoss` | `[batch, labels]` | `[batch, labels]` |
| `regression` | `RegressionLoss` | `[batch]` | `[batch]` |
| `forecasting` | `RegressionLoss` | `[batch]` | `[batch]` |

## Classification

Binary and multilabel tasks use binary cross entropy with logits. Multiclass
tasks use cross entropy over mutually exclusive class logits. Pass raw logits
to loss modules; do not apply sigmoid or softmax first.

## Regression and Forecasting

`RegressionLoss` uses Huber loss and accepts a configurable `delta`. Normalize
targets during data preparation when their scale is large or highly skewed,
and store the inverse transform with the trained artifact.

## Custom Loss Functions

A custom loss is any `torch.nn.Module` accepting predictions and labels. Use
the model's `target_features` output when the task requires a custom head, or
`target_scores` when adapting the published scalar decoder.

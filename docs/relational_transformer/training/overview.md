# Training overview

There are two adaptation paths, and both consume
`RelationalExample(input=batch, label=...)` over the same inference tensor contract. For
a raw cell-vector array, set `target=0` (or the relevant position) on the example; a
typed `RelationalBatch` already contains its target mask.

## Head Tuning

Head tuning encodes every training context once, freezes RT-J, and trains a small linear
classifier or regressor over the 512-wide target feature. It supports binary, multiclass,
multilabel, regression, and forecasting heads and leaves the backbone untouched, so
several heads can share one loaded model. Reach for it
when the pretrained representation already separates your classes and only the decision
layer is missing. See [Task-head tuning](head_tuning.md).

## Full Fine-tuning

Full fine-tuning differentiates through value encoders, relational attention blocks,
normalization, and decoder heads. The trainer optimizes the scalar target decoder, which
serves binary and regression objectives directly. It needs more data and more care than
head tuning, and it is the only path that adapts the model to a new embedding space or a
schema distribution the checkpoint has never seen. See
[Full-model fine-tuning](full_finetuning.md).

## Choosing Between Them

Fit a head first and evaluate it. When the head's metrics plateau below what the task
needs, the backbone's representation is the limit, and full fine-tuning becomes worth its
cost. Split data by time before constructing examples in either path, so future context
stays out of training features and normalization statistics.

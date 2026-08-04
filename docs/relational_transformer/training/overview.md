# Choosing a Training Path

Both adaptation paths consume `RelationalExample(input=batch, label=...)` over the same
inference tensor contract. The [Training Overview](../training_overview.md) walks through
the components every run combines; this page helps you pick a path.

## Head Tuning

Head tuning freezes RT-J, encodes each example once, and trains a small linear head over
the 512-wide target feature. Several heads can share one loaded model. See
[Task-head tuning](head_tuning.md).

## Full Fine-tuning

Full fine-tuning differentiates through value encoders, relational attention blocks,
normalization, and decoder heads. It is the path that adapts the model to a new embedding
space or a schema distribution the checkpoint has never seen. See
[Full-model fine-tuning](full_finetuning.md).

## Recommendation

Fit a head first and evaluate it. When the head's metrics plateau below what the task
needs, the backbone's representation is the limit, and full fine-tuning becomes worth its
cost. Split data by time before constructing examples in either path, so future context
stays out of training features and normalization statistics.

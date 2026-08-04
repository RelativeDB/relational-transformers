# Training Examples

Every workflow below has a complete runnable script in the
[examples directory](https://github.com/RelativeDB/relational-transformers/tree/main/examples).

## Head Tuning

[Task-head tuning](head_tuning.md) documents the workflow, and
[tune_issue_head.py](https://github.com/RelativeDB/relational-transformers/blob/main/examples/tune_issue_head.py)
runs it end to end over a frozen backbone.

## Multiclass Classification

Set `num_labels` and use `problem_type="multiclass"`. Predictions from the named head
apply softmax and return one distribution per context:

```python
head = model.fit_head(
    labeled_issues,
    task="issue_label",
    num_labels=5,
    problem_type="multiclass",
)
distribution = model.predict(batch, task_head="issue_label")
predicted_class = int(distribution.argmax())
```

## Full-model Fine-tuning

Use `RelationalTrainer` to update the backbone and scalar decoder together. Gradient
accumulation supports effective batches larger than device memory:

```python
args = RelationalTrainingArguments(
    output_dir="models/churn",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,   # optimizer sees 32 examples per step
)
RelationalTrainer(model=model, args=args, train_dataset=examples).train()
```

Script: [finetune_churn.py](https://github.com/RelativeDB/relational-transformers/blob/main/examples/finetune_churn.py)

## Evaluation During Training

Combine several evaluators so one call reports every validation metric:

```python
from relational_transformers import (
    BinaryClassificationEvaluator,
    RegressionEvaluator,
    SequentialEvaluator,
)

evaluator = SequentialEvaluator([
    BinaryClassificationEvaluator(churn_examples),
    RegressionEvaluator(spend_examples),
])
metrics = evaluator(model)
```

`AblationEvaluator` from the `relational-transformers-utils` package composes the same
way and adds named context-ablation deltas.

Script: [evaluate_churn.py](https://github.com/RelativeDB/relational-transformers/blob/main/examples/evaluate_churn.py)

## Custom PyTorch Loop

Call `model.forward(batch, output="target_features")` to train an arbitrary head, or
access `model.model` for complete control over optimization and mixed precision:

```python
import torch

backbone = model.model
optimizer = torch.optim.AdamW(backbone.parameters(), lr=1e-5)

backbone.train()
for batch, labels in loader:
    logits = backbone(batch.to(model.device), output="target_scores").scores
    loss = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

The public loss modules from the [Loss Overview](../loss_overview.md) plug into this loop
unchanged.

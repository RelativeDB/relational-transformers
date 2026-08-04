# Training Examples

## Head Tuning

Fit a binary churn head without changing the backbone:

```python
head = model.fit_head(
    training_examples,
    task="churn",
    problem_type="binary",
    epochs=100,
    learning_rate=1e-3,
)
head.save_pretrained("models/churn-head")
probabilities = model.predict(contexts, task_head="churn")
```

## Multiclass Classification

Set `num_labels` and use `problem_type="multiclass"`. Predictions from the
named head use softmax and return one distribution per context.

## Full-model Fine-tuning

Use `RelationalTrainer` to update the backbone and scalar decoder together.
Gradient accumulation supports effective batches larger than device memory.

## Custom PyTorch Loop

Call `model.forward(batch, output="target_features")` to train an arbitrary
head, or access `model.model` for complete control over optimization and mixed
precision.

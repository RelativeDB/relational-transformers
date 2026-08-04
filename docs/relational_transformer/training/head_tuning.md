# Task-head tuning

```python
examples = [
    RelationalExample(input=batch_a, label=2),
    RelationalExample(input=batch_b, label=0),
]

head = model.fit_head(
    examples,
    task="issue_label",
    num_labels=5,
    problem_type="multiclass",
)
head.save_pretrained("issue-label-head")
```

The backbone is run under inference mode and is never updated. Supported
problem types are binary, multiclass, multilabel, regression, and forecasting.

# Task-head tuning

`fit_head` adapts the model to a new task without touching the backbone. Each example is
encoded once under inference mode, then a single linear layer trains over the resulting
`[512]` target features.

```python
from relational_transformers import RelationalExample, RelationalTransformer

model = RelationalTransformer("RelativeDB/rt-j-fp16")

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
head.save_pretrained("models/issue-label-head")

distribution = model.predict(new_batch, task_head="issue_label")
```

When an example's input is a raw `[cells, 2*d_text]` vector array instead of a
`RelationalBatch`, pass its target position explicitly:

```python
example = RelationalExample(input=issue_cells, label=2, target=0)
```

The fitted head registers under `model.heads["issue_label"]`, and `predict` routes
through it whenever `task_head` names it. Supported problem types are `binary`,
`multiclass`, `multilabel`, `regression`, and `forecasting`. Head fitting requires the
torch backend.

## Tuning Knobs

`fit_head` accepts `epochs` (default 100), `learning_rate` (default `1e-3`), and
`weight_decay` (default `1e-4`). The optimizer is AdamW over the head's parameters only,
and every epoch is a full-batch step over the pre-encoded features, which is why hundreds
of epochs finish quickly. Because features are encoded once up front, the cost of a
fitting run is one forward pass per example plus a small optimization loop.

## Reloading a Saved Head

`TaskHead.save_pretrained` writes `head.safetensors` and `head_config.json`. Reattach a
saved head to any model with the same `d_model`:

```python
from relational_transformers import TaskHead

model.heads["issue_label"] = TaskHead.from_pretrained("models/issue-label-head")
```

Several heads can coexist on one loaded model, each under its own task name, so one
serving process can answer multiple prediction tasks over the same contexts.

## Fitting Over Precomputed Features

Integrations that already hold `[N, 512]` target features can skip the
example loop and fit directly with `fit_feature_head`. It standardizes
features per dimension, supports `binary`, `regression`, `multiclass`, and
grouped `ranking` heads, and can seed a multiclass head from the
checkpoint's own class-embedding basis so training starts at the zero-shot
ordering:

```python
from relational_transformers import FineTunedHead, fit_feature_head

head = fit_feature_head(
    features, labels, "multiclass", classes=classes,
    class_embeddings=normalized_label_embeddings,
    text_decoder=model.model.dec_dict["text"],
)
head.save("models/issue-head.safetensors")
logits = FineTunedHead.load("models/issue-head.safetensors").predict(features)
```

The saved artifact pairs the safetensors weights with a `.preproc.json`
sidecar carrying the feature standardization, the fitted classes, and any
preprocessing statistics the caller attached; `load` refuses a head whose
sidecar is missing, since serving it would scale every input wrongly.

# Prediction

`RelationalTransformer.predict` returns sigmoid probabilities for a classification
checkpoint and raw normalized values for a regression checkpoint. Pass
`activation="identity"` to obtain classification logits.

```python
from relational_transformers import RelationalTransformer

classifier = RelationalTransformer("RelativeDB/rt-j-fp16")
probabilities = classifier.predict(batch)

regressor = RelationalTransformer("RelativeDB/rt-j-fp16", task="regression")
normalized_values = regressor.predict(batch)
```

## Activations

`predict` picks a default activation from the problem type and lets you override it:

| Problem type | Default activation | Override |
| --- | --- | --- |
| `classification`, `binary`, `multilabel`, `clf` | sigmoid | `activation="identity"` for logits |
| `multiclass` (fitted task heads) | softmax | `activation="identity"` for logits |
| `regression`, `forecasting` | identity | none needed |

Ranking pipelines usually want logits, since sigmoid preserves order and calibration
tooling expects the raw score. Set `convert_to_numpy=False` to keep results as torch
tensors on the model's device.

## Predicting with a Fitted Task Head

After `fit_head` stores a head under a task name, route predictions through it:

```python
model.fit_head(examples, task="issue_label", num_labels=5, problem_type="multiclass")
distribution = model.predict(batch, task_head="issue_label")
```

The head runs over frozen target features. The published scalar decoder stays available
by omitting `task_head`.

## Contextual Cell Embeddings

`model.encode(context)` returns one contextualized state per cell, shaped
`[batch, cells, 512]` for the published models. Pass
`output_value="target_features"` for the summed `[batch, 512]` target representation
that task heads train on.

```python
states = model.encode(batch)                                    # [B, S, 512]
features = model.encode(batch, output_value="target_features")  # [B, 512]
```

## Model Outputs

Use `forward` when a consumer needs a lower-level output than `predict` exposes:

```python
output = model.forward(batch, output="target_features")
features = output.features    # [batch, 512]
```

| `output` | Field on `ModelOutput` | Shape | Meaning |
| --- | --- | --- | --- |
| `target_scores` | `scores` | `[batch]` | one scalar per context from the number decoder |
| `token_scores` | `token_scores` | `[batch, cells]` | the number-decoder score at every cell, in caller order |
| `target_features` | `features` | `[batch, 512]` | summed target state used for head tuning |
| `target_scores_and_text` | `scores`, `target_text` | `[batch]`, `[batch, 384]` | scalar scores plus the raw text-decoder output at the target |
| `embeddings` | `embeddings` | `[batch, cells, 512]` | contextualized state for every cell |

Every field comes back as a torch tensor on all backends. `target_text` is the raw
decoder output; nothing normalizes it, so downstream nearest-neighbor lookups decide
their own similarity convention.

```{eval-rst}
.. note::

   The ONNX and Triton backends serve ``target_scores`` only. Requesting any other
   output on those backends raises :class:`ValueError`, and the meta backend raises on
   any inference call because it has no weights.
```

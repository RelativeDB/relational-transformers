# Ablation

Ablation answers one question about a prediction: how much did this part of the context
matter? The workflow is explicit. Choose cells, create a modified batch, and compare the
same model's outputs.

```python
without_comments = batch.ablate(comment_positions)
full, ablated = model.predict([batch, without_comments], activation="identity")
delta = ablated - full
```

A large delta means the removed cells were load-bearing context for this prediction. A
delta near zero means the model reached the same score without them. Compare logits;
sigmoid saturation can hide real movement in probability space.

## How Ablation Works

`RelationalBatch.ablate` marks the selected positions as padding and leaves them in
place. Positions stay stable, node identities and foreign-key references need no remap,
and the returned batch is a copy, so the original stays usable. Attention masks exclude
padded cells entirely, which makes an ablated cell invisible to the model.

Target cells cannot be ablated; attempting it raises `ValueError`.

## Choosing What to Ablate

Positions are yours to group. Applications typically group by table (all support
tickets), by column (every `amount`), by entity (one specific order), or by time range.
The model never guesses which ablation is meaningful, so the grouping should map to a
question someone actually asked.

```python
support_positions = [i for i, table in enumerate(tables) if table == "support_tickets"]
without_support = batch.ablate(support_positions)
```

## Measuring Over a Dataset

`AblationEvaluator` runs named ablations across a set of examples and reports the mean
and mean absolute prediction delta per group:

```python
from relational_transformers import AblationEvaluator

evaluator = AblationEvaluator(
    examples,
    ablations={"support_history": support_positions, "orders": order_positions},
)
metrics = evaluator(model)
# => {'support_history_mean_delta': ..., 'support_history_mean_absolute_delta': ..., ...}
```

Deltas are computed on identity-activation outputs, so classification results are in
logit space. The
[support-history ablation example](https://github.com/RelativeDB/relational-transformers/blob/main/examples/ablate_support_history.py)
runs this end to end on typed customer contexts.

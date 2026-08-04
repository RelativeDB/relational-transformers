# Ablation

Ablation is explicit: choose cells, create a modified batch, and compare the
same model's outputs.

```python
without_comments = batch.ablate(comment_positions)
full, ablated = model.predict([batch, without_comments])
delta = ablated - full
```

`RelationalBatch.ablate` pads selected positions instead of renumbering them,
so node identities and foreign-key references stay stable. It refuses to
remove targets. Applications can group positions by table, column, time range,
or any domain concept; the model does not guess which ablation is meaningful.

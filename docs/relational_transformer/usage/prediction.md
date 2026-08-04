# Prediction

`RelationalTransformer.predict` returns sigmoid probabilities for a
classification checkpoint and raw normalized values for a regression
checkpoint. Pass `activation="identity"` to obtain classification logits.

```python
classifier = RelationalTransformer("RelativeDB/rt-j-fp16")
probabilities = classifier.predict(batch)

regressor = RelationalTransformer("RelativeDB/rt-j-fp16", task="regression")
normalized_values = regressor.predict(batch)
```

Use `forward` when a consumer needs a lower-level output:

- `target_scores`: one scalar per context;
- `token_scores`: one number-head score per cell;
- `target_features`: the 512-wide target state used for head tuning;
- `target_scores_and_text`: scalar scores plus the 384-wide text decoder;
- `embeddings`: contextualized state for every cell.

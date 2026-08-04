# Computing Predictions

## Prediction

`model.predict()` accepts a `RelationalBatch`, a canonical tensor mapping, a
simple all-text cell-vector array, or a sequence of one-context inputs.

## Contextual Cell Embeddings

`model.encode(context)` returns one contextual state per cell. Use
`output_value="target_features"` to return the summed target representation
used by task heads.

## Classification and Regression

Classification applies sigmoid by default. Multiclass task heads apply
softmax. Regression and forecasting return raw scalar outputs. Override with
`activation="identity"` whenever calibration or ranking consumes logits.

## Batching

Variable-length contexts are padded during sequence collation. All contexts
must use the same text embedding width and model contract.

## Ablation

Call `batch.ablate(positions)` to mask caller-selected cells without changing
stable node or parent IDs. Compare the resulting prediction with the original;
the library never chooses an ablation on the user's behalf.

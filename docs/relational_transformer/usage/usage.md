# Computing Predictions

Characteristics of a Relational Transformer model:

1. Predicts a **masked target cell** from a context of related, typed cells.
2. Consumes **embeddings your application already created**; the library never encodes strings.
3. Supports classification, regression, forecasting, multilabel ranking, and custom heads over the same context.
4. Runs the same input contract on PyTorch, Triton CUDA, and ONNX Runtime.

## Prediction

`model.predict()` accepts a `RelationalBatch`, a mapping of canonical tensor fields, a
simple all-text cell-vector array, or a sequence of single-context inputs.

```{eval-rst}
.. sidebar:: Documentation

   #. :meth:`RelationalTransformer.predict() <relational_transformers.RelationalTransformer.predict>`
   #. :meth:`RelationalTransformer.encode() <relational_transformers.RelationalTransformer.encode>`
   #. :class:`~relational_transformers.RelationalBatch`
```

```python
from relational_transformers import RelationalTransformer

model = RelationalTransformer("RelativeDB/rt-j-fp16")

probability = model.predict(batch)                       # one RelationalBatch
probabilities = model.predict([cells_a, cells_b], target=0)  # list of cell arrays
```

A single context returns a Python float. Batched inputs return a numpy array with one
value per context. The [Prediction](prediction.md) page lists every output the model can
produce below this convenience layer.

## Contextual Cell Embeddings

`model.encode(context)` returns one contextualized state per cell, shaped
`[batch, cells, 512]` for the published models. Pass
`output_value="target_features"` to get the summed target representation instead, which
is the `[batch, 512]` feature that task heads train on.

```python
states = model.encode(batch)                                  # [B, S, 512]
features = model.encode(batch, output_value="target_features")  # [B, 512]
```

## Classification and Regression

Classification checkpoints apply a sigmoid inside `predict` by default. Multiclass task
heads apply softmax. Regression and forecasting return raw values in the checkpoint's
normalized target space. Override the default with `activation="identity"` whenever
calibration or ranking needs the underlying logit.

```python
logit = model.predict(batch, activation="identity")
```

## Batching

Variable-length contexts are padded during sequence collation. Every context in a batch
must share the same text embedding width and checkpoint contract. Padded cells are
masked out of attention.

## Ablation

`batch.ablate(positions)` masks caller-selected cells and keeps node and parent IDs
stable, so you can compare a prediction with and without part of its context. The
[Ablation](ablation.md) page shows the full workflow, and `AblationEvaluator` turns it
into a metric over a dataset.

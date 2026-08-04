# Computing Predictions

Characteristics of a Relational Transformer model:

1. Predicts a **masked target cell** from a context of related, typed cells.
2. Consumes **embeddings your application already created**; the library never encodes strings.
3. Supports classification, regression, forecasting, multilabel ranking, and custom heads over the same context.
4. Runs the same input contract on PyTorch, Triton CUDA, and ONNX Runtime.

`model.predict()` accepts a `RelationalBatch`, a mapping of canonical tensor fields, a
simple all-text cell-vector array, or a sequence of single-context inputs. A single
context returns a Python float. Batched inputs return a numpy array with one value per
context, padded and collated for you.

```python
from relational_transformers import RelationalTransformer

model = RelationalTransformer("RelativeDB/rt-j-fp16")

probability = model.predict(batch)                           # one RelationalBatch
probabilities = model.predict([cells_a, cells_b], target=0)  # list of cell arrays
```

Each topic has its own page:

- [Prediction](prediction.md): activations, logits, task heads, cell embeddings, and every model output.
- [Relational Batches](batches.md): the tensor contract each backend consumes.
- [Backends](backends.md): PyTorch, Triton, ONNX, and meta loading.
- [Ablation](ablation.md): measuring how much a group of cells affects a prediction.
- [Efficiency](efficiency.md): batching by length, compilation, and quantized checkpoints.

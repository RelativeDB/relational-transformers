---
license: apache-2.0
library_name: relational-transformers
pipeline_tag: tabular-classification
tags:
  - relational-transformers
  - onnx
  - relational-data
---

# RT-J ONNX

This repository contains the classification checkpoint from
[`RelativeDB/rt-j-fp16`](https://huggingface.co/RelativeDB/rt-j-fp16), exported
to ONNX for framework-neutral target prediction over caller-provided relational
cell embeddings.

Load and cache it automatically with:

```python
from relational_transformers import RelationalTransformer

model = RelationalTransformer(backend="onnx")
predictions = model.predict(batch)
```

The graph accepts the canonical `RelationalBatch` tensor fields. Batch size and
cell count are dynamic; the text and column-embedding width is fixed at 384.
Callers remain responsible for producing the model-ready cell embeddings and
relations described in the
[`relational-transformers` input contract](https://relationaltransformers.com/docs/relational_transformer/usage/batches.html).

`model.onnx` is exported from the full published checkpoint. The release
process verifies PyTorch and ONNX Runtime output parity at multiple dynamic
context lengths before publishing the file.

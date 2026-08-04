# Testing

The default suite uses small deterministic RT-J checkpoints and realistic
customer contexts containing typed customer, order, and support-ticket cells.
It does not need network access or a GPU.

```bash
pytest
```

The portable coverage gate excludes CUDA-only Triton kernel implementations,
which cannot execute on CPU CI, and requires at least 90% coverage. The current
portable suite reports over 97%:

```bash
make coverage
```

The suite covers:

- typed relational batches, wide production-style node IDs, foreign-key parents, and padding;
- classification, regression, batching, output views, and explicit support-ticket ablation;
- checkpoint save/reload plus int8 and packed-int4 dequantization;
- frozen-backbone binary and multiclass head tuning and full-model fine-tuning;
- dynamic-batch and dynamic-context ONNX export and inference;
- Triton sorting and relational attention work-list construction;
- malformed shapes, missing fields, invalid semantic types, and non-finite values;
- meta-device architecture inspection without parameter allocation.

Published-model tests are opt-in because they download large checkpoints:

```bash
RUN_HUB_TESTS=1 pytest -m hub
```

On a CUDA deployment host, compare the optimized Triton output directly with
the PyTorch backend on the same relational context:

```bash
RUN_CUDA_TESTS=1 pytest -m cuda
```

The RelativeDB repository has an additional end-to-end integration test. It
runs a real `PREDICT` query through retrieval, cell encoding, relational batch
construction, and this package's PyTorch runtime.

# Testing

The default suite uses small deterministic RT-J checkpoints and realistic customer
contexts containing typed customer, order, and support-ticket cells. It runs offline on
CPU and finishes in a few minutes.

```bash
pytest
```

## Coverage

The portable coverage gate requires at least 90% and excludes the CUDA-only Triton kernel
implementations, which cannot execute on CPU CI. The current portable suite reports over
97%:

```bash
make coverage
```

```{eval-rst}
.. collapse:: What the default suite covers

   - typed relational batches, wide production-style node IDs, foreign-key parents, and padding;
   - classification, regression, batching, output views, and explicit support-ticket ablation;
   - checkpoint save/reload plus int8 and packed-int4 dequantization;
   - frozen-backbone binary and multiclass head tuning and full-model fine-tuning;
   - dynamic-batch and dynamic-context ONNX export and inference;
   - Triton sorting and relational attention work-list construction;
   - malformed shapes, missing fields, invalid semantic types, and non-finite values;
   - meta-device architecture inspection without parameter allocation.
```

## Opt-in Suites

Published-model tests download large checkpoints from the Hugging Face Hub, so they only
run when requested:

```bash
RUN_HUB_TESTS=1 pytest -m hub
```

On a CUDA deployment host, compare the optimized Triton output directly with the PyTorch
backend on the same relational context:

```bash
RUN_CUDA_TESTS=1 pytest -m cuda
```

## Linting and Documentation Checks

```bash
make lint      # ruff over the package and tests
make docs      # strict Sphinx build; any warning fails
```

## Integration Testing

The RelativeDB repository has an additional end-to-end integration test. It runs a real
`PREDICT` query through retrieval, cell encoding, relational batch construction, and this
package's PyTorch runtime.

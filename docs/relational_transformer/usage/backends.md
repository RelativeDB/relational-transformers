# Backends

One constructor selects among four backends. All of them consume the same
`RelationalBatch` contract.

| Backend | Devices | Outputs | Training | Extra dependency |
| --- | --- | --- | --- | --- |
| `torch` (default) | CPU, MPS, CUDA | all | yes | none |
| `triton` | CUDA | `target_scores` | no | `[triton]` |
| `onnx` | ONNX Runtime providers | `target_scores` | no | `[onnx]` |
| `meta` | none | none | no | none |

## PyTorch

The default backend uses a dense, weight-compatible RT-J implementation. It supports
autograd, target features, and every decoder output, which makes it the required backend
for head tuning, full fine-tuning, checkpoint saving, and ONNX export.

```python
model = RelationalTransformer("RelativeDB/rt-j-fp16", backend="torch", device="mps")
```

Weights keep their checkpoint dtype on accelerators and are widened to float32 on CPU.
Pass `compile=True` to wrap the module in `torch.compile` with dynamic shapes.

```{eval-rst}
.. collapse:: Legacy attention checkpoints

   Current RT-J checkpoints carry gated attention with per-head scale parameters. The
   loader inspects the state dict, and when the gate weights are absent it constructs the
   older ungated attention with standard scaling. Both variants share every other
   parameter name, so old and new checkpoints load through the same constructor.
```

## Triton

The Triton backend runs packed relational work lists through custom CUDA kernels. It
serves scalar classification and regression target scores.

```python
model = RelationalTransformer("RelativeDB/rt-j-fp16", backend="triton")
scores = model.predict(batch)

regressor = RelationalTransformer(
    "RelativeDB/rt-j-fp16", backend="triton", task="regression"
)
```

FP8 checkpoints can stay in FP8 here. Validate a Triton deployment against the PyTorch
backend on the same contexts with `RUN_CUDA_TESTS=1 pytest -m cuda`.

## ONNX

Load the published ONNX graph directly; the first use downloads and caches
`RelativeDB/rt-j-onnx` through the Hugging Face Hub.

```python
served = RelationalTransformer(backend="onnx")
scores = served.predict(batch)
```

Export requires the PyTorch backend and one example batch. The example fixes `d_text` and
the architecture; batch size and context length stay dynamic in the exported graph.

```python
model.export_onnx("rt-j.onnx", example_batch)

served = RelationalTransformer("rt-j.onnx", backend="onnx")
scores = served.predict(batch)
```

An ONNX argument may also be a Hub repository containing `model.onnx`, a local directory
containing that file, or the file itself. The backend takes an optional `providers` list, which passes straight through to
`onnxruntime.InferenceSession`. Predictions come back as torch tensors, matching the
other backends. A host using this package installs the package and its `[onnx]` extra; a
custom host calling ONNX Runtime directly can serve the exported file with ONNX Runtime
and the matching input contract.

## Meta

Meta construction downloads only the configuration and allocates no parameter storage.
Use it to inspect modules, count parameters, or plan sharding for a model too large to
load.

```python
meta = RelationalTransformer("RelativeDB/rt-j-fp16", backend="meta")
print(meta.get_model_kwargs())
# => {'num_blocks': 12, 'd_model': 512, 'd_text': 384, 'num_heads': 8, 'd_ff': 2048}
```

Any inference call on a meta model raises `RuntimeError`, since there are no weights to
run.

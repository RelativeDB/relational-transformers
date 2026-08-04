# Backends

## PyTorch

The default backend uses a dense, weight-compatible RT-J implementation. It
supports CPU, MPS, CUDA, autograd, target features, and every decoder output.

```python
model = RelationalTransformer(model_id, backend="torch", device="mps")
```
## Triton

The Triton backend uses packed relational work lists and custom CUDA kernels.
It is intended for classification target-score serving.

```python
model = RelationalTransformer(model_id, backend="triton")
```

## ONNX

Export requires the PyTorch backend and an example batch. Batch and sequence
axes are dynamic; embedding width and the model architecture are fixed.

```python
model.export_onnx("rt-j.onnx", example_batch)
served = RelationalTransformer("rt-j.onnx", backend="onnx")
```

## Meta

Meta construction downloads only configuration and allocates no parameter
storage. It is useful for module inspection, parameter counting, planning, and
transformation code.

```python
meta = RelationalTransformer(model_id, backend="meta")
print(meta.get_model_kwargs())
```

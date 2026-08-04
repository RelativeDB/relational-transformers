# ONNX API

`RelationalTransformer.export_onnx` and the `onnx` backend wrap these directly; the
[Backends](../relational_transformer/usage/backends.md#onnx) page shows the export and
serving workflow.

When no model is specified, `RelationalTransformer(backend="onnx")` downloads
`RelativeDB/rt-j-onnx/model.onnx` automatically.

## export_onnx

```{eval-rst}
.. autofunction:: relational_transformers.onnx.export_onnx
```

## OnnxBackend

```{eval-rst}
.. autoclass:: relational_transformers.onnx.OnnxBackend
   :members:
```

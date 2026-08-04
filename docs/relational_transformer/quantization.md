# Quantization

## FP8

FP8 checkpoints store two-dimensional model weights as `float8_e4m3fn` and
retain biases, normalization scales, and mask embeddings in their original
floating-point type. They load through the normal constructor and can remain
FP8 in the Triton CUDA backend.

```bash
relational-transformers-quantize \
  RelativeDB/rt-j-fp16 RelativeDB/rt-j-fp8-local
```

## Int8

Int8 checkpoints use one floating-point scale per output row. Portable PyTorch
dequantizes them during loading. RelativeDB's native engine can operate on the
quantized representation directly.

## Int4

Int4 checkpoints pack two weights per byte in groups of 32 and store a scale
and minimum per group. They provide the smallest artifact and the largest
accuracy tradeoff.

## Validation

Always compare logits, ranking, and task metrics against FP16 on representative
relational contexts. The test suite validates every published format when
`RUN_HUB_TESTS=1` and compares Triton with PyTorch when `RUN_CUDA_TESTS=1`.

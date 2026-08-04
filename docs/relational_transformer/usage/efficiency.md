# Efficiency

## Backend Selection

Use PyTorch for portability and training, Triton for optimized CUDA serving,
and ONNX Runtime for framework-neutral deployment. Meta models inspect large
architectures without allocating parameter storage.

## Batch Similar Context Lengths

Padding cost grows with the longest context in a batch. Group contexts of
similar lengths when throughput matters, while preserving one target and its
complete relational neighborhood per batch row.

## Quantized Models

FP8 and integer checkpoints reduce storage and transfer costs. FP8 remains a
native floating-point weight format for supported CUDA hardware. Portable
PyTorch expands int8 and int4 checkpoints while loading; the native RelativeDB
engine can keep those formats packed during inference.

## ONNX Export

ONNX exports have dynamic batch and cell axes but a fixed architecture and
embedding width. Export once from the exact checkpoint used in validation.

## Measure End-to-End Latency

Include retrieval, application-owned cell encoding, collation, model execution,
and output calibration in deployment benchmarks. Model-only timing can hide the
dominant cost for small contexts.

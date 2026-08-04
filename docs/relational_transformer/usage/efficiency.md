# Efficiency

## Backend Selection

Use PyTorch for portability and training, Triton for optimized CUDA serving, and ONNX
Runtime for framework-neutral deployment. Meta models inspect large architectures without
allocating parameter storage. The [Backends](backends.md) page compares capabilities;
this page is about making whichever backend you chose fast.

On the PyTorch backend, `compile=True` wraps the model in `torch.compile` with dynamic
shapes. Compilation pays off on a serving process that handles many batches and costs a
warm-up on the first few calls.

## Batch Similar Context Lengths

Collation pads every context in a batch to the longest one, and relational attention
works over `[cells, cells]` masks, so cost grows quadratically with the padded length.
One 400-cell context in a batch of 50-cell contexts makes the whole batch pay 400-cell
prices. Group contexts of similar lengths when throughput matters, while preserving one
target and its complete relational neighborhood per batch row.

## Quantized Models

Published FP8 and integer checkpoints cut storage and transfer costs. FP8 remains a
native floating-point weight format for supported CUDA hardware and stays packed in the
Triton backend. Portable PyTorch expands int8 and int4 checkpoints to full precision
while loading, so those formats save disk and download time but not runtime memory. This
package loads those formats; checkpoint creation belongs to the deployment pipeline.
Always compare task metrics against `RelativeDB/rt-j-fp16` before switching formats.

## ONNX Export

ONNX exports have dynamic batch and cell axes over a fixed architecture and embedding
width. Export once from the exact checkpoint used in validation, then version the file
alongside the config that produced it. Provider selection happens at load time through
the `providers` argument, so one exported file serves CPU and GPU hosts.

## Measure End-to-End Latency

Include retrieval, application-owned cell encoding, collation, model execution, and
output calibration in deployment benchmarks. Model-only timing can hide the dominant cost
for small contexts, where text encoding often takes longer than the forward pass.

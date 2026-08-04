# Pretrained Models

RelativeDB publishes RT-J checkpoints on the Hugging Face Hub. Each repository contains
`classification/` and `regression/` subfolders with the same architecture and input
contract, so one integration serves both task families.

```python
from relational_transformers import RelationalTransformer

classifier = RelationalTransformer("RelativeDB/rt-j-fp16")
regressor = RelationalTransformer("RelativeDB/rt-j-fp16", task="regression")
```

## Published Models

| Model | Storage | Portable PyTorch | Triton CUDA | Intended use |
| --- | ---: | :---: | :---: | --- |
| [`RelativeDB/rt-j-fp16`](https://huggingface.co/RelativeDB/rt-j-fp16) | 16-bit | ✓ | ✓ | Highest-fidelity inference and fine-tuning |
| [`RelativeDB/rt-j-fp8`](https://huggingface.co/RelativeDB/rt-j-fp8) | 8-bit float | ✓ | ✓ | CUDA deployment with native FP8 weights |
| [`RelativeDB/rt-j-int8`](https://huggingface.co/RelativeDB/rt-j-int8) | 8-bit integer | ✓ | — | Compact portable inference |
| [`RelativeDB/rt-j-int4`](https://huggingface.co/RelativeDB/rt-j-int4) | packed 4-bit | ✓ | — | Smallest portable checkpoint |

Start with `rt-j-fp16`. It is the default checkpoint, the fine-tuning base, and the
full-precision baseline to use when validating a deployment format.

## Model Input Contract

The released models use 384-wide text and column embeddings from
`sentence-transformers/all-MiniLM-L12-v2`, 512-wide hidden states, 12 blocks, 8 attention
heads, and a 2048-wide feed-forward layer.

Matching only the embedding width is insufficient. A checkpoint is trained against one
embedding space, one normalization scheme for scalars and timestamps, and the semantic
type, target masking, and relation conventions described in
[Relational Batches](usage/batches.md). Callers must preserve all of them, and each model
card records what its checkpoint expects. For a different encoder, see
[Changing the Embedding Space](usage/custom_models.md#changing-the-embedding-space).

## Classification and Ranking

The classification checkpoint emits a raw target logit. `predict()` applies a sigmoid by
default. Request `activation="identity"` for ranking or calibration workflows that need
the original logit; the sigmoid preserves ranking order, and calibration libraries expect
the raw score.

Multiclass and multilabel tasks run through fitted task heads over the frozen backbone.
The [head tuning](training/head_tuning.md) page shows the workflow.

## Regression and Forecasting

Select `task="regression"` for scalar regression and forecasting. Predictions come back
in the checkpoint's normalized target space, so the application applies the inverse
transform associated with its training statistics. Store that transform with the model
artifact; a prediction without its denormalization is a bare number with no unit.

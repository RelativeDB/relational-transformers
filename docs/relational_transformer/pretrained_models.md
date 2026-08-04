# Pretrained Models

RelativeDB publishes RT-J checkpoints on the Hugging Face Hub. Each repository
contains `classification/` and `regression/` subfolders with the same model
architecture and input contract.

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

## Model Input Contract

The released models use 384-wide text and column embeddings from
`sentence-transformers/all-MiniLM-L12-v2`, 512-wide hidden states, 12 blocks,
8 attention heads, and a 2048-wide feed-forward layer. Matching only the
embedding width is insufficient: callers must preserve the encoder,
normalization, semantic type, target masking, and relation conventions.

## Classification and Ranking

The classification checkpoint emits a raw target logit. `predict()` applies a
sigmoid by default. Request `activation="identity"` for ranking or calibration
workflows that need the original logit.

## Regression and Forecasting

Select `task="regression"` for scalar regression and forecasting. Predictions
are returned in the checkpoint's normalized target space; applications must
apply the inverse transform associated with their training statistics.

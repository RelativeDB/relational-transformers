# Examples

These examples begin with application-owned values and embeddings, then pass
model-ready cell vectors or a typed `RelationalBatch` to the transformer. They
do not hide parsing or encoding inside the model API.

Install the example dependencies first:

```bash
python -m pip install -e ".[dev,onnx]" sentence-transformers
```

## Prediction and analysis

- `predict_issue.py` — encode issue fields and make one classification prediction
- `batch_predictions.py` — score variable-length issue contexts in one call
- `typed_customer_churn.py` — build typed customer, order, and support tensors
- `ablate_support_history.py` — define and compare an explicit ablation
- `evaluate_churn.py` — run classification and ablation evaluators

## Training

- `tune_issue_head.py` — fit and save a multiclass head over a frozen backbone
- `finetune_churn.py` — fine-tune all RT-J weights with mini-batches

## Deployment

- `export_onnx.py` — export dynamic batch/sequence axes and verify ONNX Runtime
- `inspect_meta_model.py` — inspect model dimensions without allocating weights
- `triton_fp8_inference.py` — run native FP8 weights with Triton on CUDA
- `quantize_fp8.py` — produce classification and regression FP8 checkpoints

Run any file from this directory or from the repository root, for example:

```bash
python examples/batch_predictions.py
```

The prediction examples download `RelativeDB/rt-j-fp16` and
`sentence-transformers/all-MiniLM-L12-v2` on first use. The Triton example
requires a CUDA device and the `triton` extra.

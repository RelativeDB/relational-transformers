<!--- BADGES: START --->

[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-models-yellow)][#models]
[![GitHub - License](https://img.shields.io/github/license/RelativeDB/relational-transformers?logo=github&style=flat&color=green)][#github-license]
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/relational-transformers?logo=pypi&style=flat&color=blue)][#pypi-package]
[![PyPI - Package Version](https://img.shields.io/pypi/v/relational-transformers?logo=pypi&style=flat&color=orange)][#pypi-package]
[![Docs](https://img.shields.io/static/v1?logo=readthedocs&style=flat&color=pink&label=docs&message=relational-transformers)][#docs-package]

<!--- BADGES: END --->

# Relational Transformers: Prediction and Fine-Tuning over Related Data

This framework provides an easy method to run and train relational transformer models over user-provided embeddings. It can be used to make predictions from sets of related cells ([quickstart](https://relationaltransformers.com/docs/quickstart.html#load-rt-j)), measure which parts of a context affect those predictions ([ablation](https://relationaltransformers.com/docs/relational_transformer/usage/ablation.html)), fit lightweight task heads ([training overview](https://relationaltransformers.com/docs/relational_transformer/training/overview.html#head-tuning)), or fine-tune a complete model ([training overview](https://relationaltransformers.com/docs/relational_transformer/training/overview.html#full-fine-tuning)). This supports binary and multiclass classification, regression, forecasting, multilabel ranking, and other prediction tasks over related data.

Unlike frameworks that start with raw text or tables, Relational Transformers starts with **embeddings that you have already created**. You choose how strings, numbers, timestamps, images, categories, and domain objects become vectors. The framework handles typed relations, batching, relational attention, training, evaluation, and model checkpoints. It never silently downloads an encoding model or couples your model to a particular database.

```text
your data → your encoders → embeddings + relations → RelationalTransformer → predictions
```

Pretrained models, fitted task heads, and fine-tuned checkpoints can be shared through the Hugging Face Hub. Each model declares its required embedding space, input dimension, and relation vocabulary in its model card. You can use a model as published, fit a small head over its frozen cell states, or fine-tune the complete relational transformer for your own feature pipeline.

For the **full documentation**, see **[Relational Transformers Documentation][#docs-package]**.

## Installation

We recommend **Python 3.10+** and **[PyTorch 2.2+](https://pytorch.org/get-started/locally/)**.

```bash
pip install -U relational-transformers
```

See [Installation](https://relationaltransformers.com/docs/installation.html) in the docs for source and editable installs and the ONNX, Triton, documentation, and development extras.

## Getting Started

See [Quickstart](https://relationaltransformers.com/docs/quickstart.html) in our documentation.

### Relational Prediction Models

First load a pretrained Relational Transformer and the encoder your application uses for text-valued cells.

```python
from sentence_transformers import SentenceTransformer
from relational_transformers import RelationalTransformer

text_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
model = RelationalTransformer("RelativeDB/rt-j-fp16")
```

Suppose your application needs to classify whether a GitHub issue is a bug. RT-J text cells have two separately encoded channels: the column and its value. Concatenate those two embeddings to build each model-ready cell vector. The masked target has a column embedding and a zero value channel.

```python
import numpy as np

issue = {
    "title": "Database connections time out after 30 seconds",
    "body": "The pool stops returning connections after the service has been idle.",
    "latest_comment": "Restarting the process temporarily fixes it.",
}

def text_cell_vector(column, value):
    column_vector = text_encoder.encode(column)
    value_vector = text_encoder.encode(value)
    return np.concatenate([column_vector, value_vector])

def target_cell_vector(column):
    column_vector = text_encoder.encode(column)
    masked_value_vector = np.zeros(384, dtype=np.float32)
    return np.concatenate([column_vector, masked_value_vector])

bug_target_vector = target_cell_vector("is bug")
title_vector = text_cell_vector("title", issue["title"])
body_vector = text_cell_vector("body", issue["body"])
comment_vector = text_cell_vector("latest comment", issue["latest_comment"])

cell_vectors = np.stack([
    bug_target_vector,
    title_vector,
    body_vector,
    comment_vector,
])

probability = float(model.predict(cell_vectors, target=0))
print(f"P(issue is a bug) = {probability:.1%}")
```

And that's already it. RT-J receives only the `[column_embedding, value_embedding]` vectors; it never receives the issue dictionary or its strings. `target=0` marks `bug_target_vector` as the masked prediction target. RelativeDB will construct this same model-ready representation from its schema and retrieved context before calling this library. Pass a list of vector arrays to batch several issues. See [Encoding Cells](https://relationaltransformers.com/docs/quickstart.html#encoding-cells) for the full typed-cell contract.

### Ablation

Ablation is just the same prediction with context deliberately removed. Here we remove the comment vector, then run the full and ablated contexts together.

```python
without_comment = np.delete(cell_vectors, 3, axis=0)
full, ablated = model.predict(
    [cell_vectors, without_comment],
    target=0,
)

print(f"with latest comment:    {full:.1%}")
print(f"without latest comment: {ablated:.1%}")
print(f"change:                 {ablated - full:+.1%}")
```

A large change means the removed cell was load-bearing context; a change near zero means it was not affecting this prediction. Nothing automatically decides what to remove—you define the ablation that answers your question and compare its prediction with the original.

## Pre-Trained Models

Pretrained RT-J models are available from [RelativeDB on the Hugging Face Hub][#models]. Each repository contains both `classification/` and `regression/` checkpoints. Classification is loaded by default; select the regression checkpoint with `RelationalTransformer(..., task="regression")`.

- [`RelativeDB/rt-j-fp16`](https://huggingface.co/RelativeDB/rt-j-fp16) — half-precision weights
- [`RelativeDB/rt-j-fp8`](https://huggingface.co/RelativeDB/rt-j-fp8) — native E4M3 FP8 matrix weights
- [`RelativeDB/rt-j-int8`](https://huggingface.co/RelativeDB/rt-j-int8) — 8-bit quantized weights
- [`RelativeDB/rt-j-int4`](https://huggingface.co/RelativeDB/rt-j-int4) — 4-bit quantized weights
- [Prediction](https://relationaltransformers.com/docs/relational_transformer/usage/prediction.html)
- [Batches and the Model Input Contract](https://relationaltransformers.com/docs/relational_transformer/usage/batches.html)
- [Custom and Local Models](https://relationaltransformers.com/docs/relational_transformer/usage/custom_models.html)

The published configs specify RT-J's 384-wide text input, 512-wide hidden states, 12 transformer blocks, 8 attention heads, and expected `all-MiniLM-L12-v2` embedding space. Matching `d_text=384` alone is not an interoperability guarantee: inputs must use the embedding model, normalization, semantic conventions, and relational structure documented by the checkpoint. For a different embedding space, train an input adapter or fine-tune a checkpoint with appropriate data.

## Backends

The same constructor selects portable PyTorch, optimized Triton CUDA, ONNX Runtime, or a zero-allocation meta model.

```python
# CPU, MPS, or CUDA; supports inference and training
model = RelationalTransformer("RelativeDB/rt-j-fp16", backend="torch")

# CUDA inference through the optimized relational-attention kernels
model = RelationalTransformer("RelativeDB/rt-j-fp16", backend="triton")

# Inspect dimensions and modules without allocating 85 million parameters
model = RelationalTransformer("RelativeDB/rt-j-fp16", backend="meta")
print(model.get_model_kwargs())
```

Export a PyTorch model once and load the resulting file anywhere ONNX Runtime is available:

```python
torch_model.export_onnx("rt-j.onnx", example_batch)
onnx_model = RelationalTransformer("rt-j.onnx", backend="onnx")
predictions = onnx_model.predict(batch)
```

See [Backends](https://relationaltransformers.com/docs/relational_transformer/usage/backends.html) for supported devices, ONNX dynamic axes, and Triton limitations.

## Training

This framework allows you to adapt relational transformer models to your own feature pipeline and task. You can fit a small multiclass or multilabel-ranking head over a frozen backbone, fine-tune the complete model for scalar binary or regression tasks with `RelationalTrainer`, or use the model in an ordinary PyTorch loop.

- **Task-Head Tuning**
  - [Frozen-Backbone Head Tuning](https://relationaltransformers.com/docs/relational_transformer/training/head_tuning.html)
- **Full-Model Fine-Tuning**
  - [Full-Model Fine-Tuning](https://relationaltransformers.com/docs/relational_transformer/training/full_finetuning.html)
- **Custom Models**
  - [Custom Models and Local Checkpoints](https://relationaltransformers.com/docs/relational_transformer/usage/custom_models.html)

A frozen-backbone head is the fastest adaptation path. Each training input is encoded once, then only the selected task head is optimized.

```python
from relational_transformers import RelationalExample

head_dataset = [
    RelationalExample(input=issue_a_batch, label=2),
    RelationalExample(input=issue_b_batch, label=0),
]

model = RelationalTransformer("RelativeDB/rt-j-fp16")
head = model.fit_head(
    head_dataset,
    task="issue_label",
    num_labels=5,
    problem_type="multiclass",
    epochs=100,
    learning_rate=1e-3,
)
head.save_pretrained("models/issue-label-head")
```

Use full-model fine-tuning when the relational backbone itself must adapt:

```python
from relational_transformers import (
    RelationalExample,
    RelationalTrainer,
    RelationalTrainingArguments,
)

train_dataset = [
    RelationalExample(input=customer_a_batch, label=1.0),
    RelationalExample(input=customer_b_batch, label=0.0),
]

model = RelationalTransformer("RelativeDB/rt-j-fp16")
args = RelationalTrainingArguments(
    output_dir="models/customer-churn",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    learning_rate=2e-5,
)

trainer = RelationalTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    task="churn",
)
trainer.train()
```

Some highlights across the different types of training are:

- User-provided embeddings for text, numbers, categories, images, and other modalities
- Typed sparse relations and variable-length relational inputs
- Frozen-backbone multiclass and multilabel-ranking head tuning
- Full-model fine-tuning for scalar binary and regression tasks
- Binary, multiclass, multilabel, regression, forecasting, and ranking objectives
- Multi-task adaptation through named prediction heads
- Frozen-head tuning and full-model PyTorch fine-tuning
- Ordinary PyTorch modules and optimizers for custom training loops

## Application Examples

The [examples directory](https://github.com/RelativeDB/relational-transformers/tree/main/examples) contains complete, runnable workflows:

- [Issue prediction](https://github.com/RelativeDB/relational-transformers/blob/main/examples/predict_issue.py) builds externally encoded column/value vectors and predicts a real label.
- [Batched prediction](https://github.com/RelativeDB/relational-transformers/blob/main/examples/batch_predictions.py) scores variable-length contexts together.
- [Typed customer churn](https://github.com/RelativeDB/relational-transformers/blob/main/examples/typed_customer_churn.py) constructs scalar, text, table, node, and foreign-key tensors explicitly.
- [Support-history ablation](https://github.com/RelativeDB/relational-transformers/blob/main/examples/ablate_support_history.py) compares caller-defined contexts.
- [Evaluation](https://github.com/RelativeDB/relational-transformers/blob/main/examples/evaluate_churn.py) combines classification and ablation metrics.
- [Task-head tuning](https://github.com/RelativeDB/relational-transformers/blob/main/examples/tune_issue_head.py) trains a multiclass issue head over a frozen backbone.
- [Full fine-tuning](https://github.com/RelativeDB/relational-transformers/blob/main/examples/finetune_churn.py) adapts the complete model with mini-batches.
- [ONNX export](https://github.com/RelativeDB/relational-transformers/blob/main/examples/export_onnx.py), [meta inspection](https://github.com/RelativeDB/relational-transformers/blob/main/examples/inspect_meta_model.py), [Triton FP8 inference](https://github.com/RelativeDB/relational-transformers/blob/main/examples/triton_fp8_inference.py), and [FP8 quantization](https://github.com/RelativeDB/relational-transformers/blob/main/examples/quantize_fp8.py) cover deployment workflows.

RelativeDB is the first production integration: it retrieves related rows, constructs typed `RelationalBatch` inputs, and selects the PyTorch, Triton, ONNX, or native serving path.

## Companion Resources

- [RelativeDB models on Hugging Face](https://huggingface.co/RelativeDB)
- [RelativeDB](https://github.com/RelativeDB/RelQL), the reference retrieval and context-construction integration
- [RT-J](https://huggingface.co/stanford-star/rt-j), the upstream relational foundation model

## Development setup

After cloning the repository (or a fork), install it in editable mode with the development dependencies:

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

To test your changes, run:

```bash
pytest
```

This runs deterministic, offline tests over typed customer, order, and support
contexts. To validate the published Hugging Face checkpoints or compare Triton
with PyTorch on CUDA, see the [Testing guide](https://relationaltransformers.com/docs/testing.html).

To build the documentation, run:

```bash
make docs
```

## Citing & Authors

If you find Relational Transformers useful in your research or application, you can cite the software:

```bibtex
@software{relational_transformers_2026,
    title = {Relational Transformers: Prediction and Fine-Tuning over Related Data},
    author = {{RelativeDB}},
    year = {2026},
    url = {https://github.com/RelativeDB/relational-transformers},
}
```

Don't hesitate to open an issue if something is broken or if you have questions about using your own embedding pipeline.

### Maintainers

Relational Transformers is maintained by RelativeDB.

## License

Relational Transformers is licensed under the [Apache License 2.0][#github-license].

[#docs-package]: https://relationaltransformers.com/
[#github-license]: https://github.com/RelativeDB/relational-transformers/blob/main/LICENSE
[#models]: https://huggingface.co/RelativeDB
[#pypi-package]: https://pypi.org/project/relational-transformers/

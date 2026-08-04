# Training Overview

Relational Transformers supports frozen-backbone head tuning, complete RT-J fine-tuning,
and ordinary PyTorch loops. This page walks through the pieces that every training run
combines; the [training subpages](training/overview.md) go deeper on each path.

## Why Fine-tune?

The published checkpoints provide a reusable starting point, but they have never seen
your schema, target definition, or encoder. Fine-tuning adapts
the model to a new cell encoder, schema distribution, target, or domain. Start with a
frozen task head when the published backbone already produces useful target features;
update the full model only when the relational representation itself must change. Head
tuning encodes each example once and then optimizes only the selected head.

## Training Components

<div class="components">
    <a href="#model" class="box">
        <div class="header">Model</div>
        Load a trainable PyTorch model.
    </a>
    <a href="#dataset" class="box">
        <div class="header">Dataset</div>
        Pair contexts with labels.
    </a>
    <a href="#loss-function" class="box">
        <div class="header">Loss</div>
        Pick the loss for your problem type.
    </a>
    <a href="#training-arguments" class="box optional">
        <div class="header">Arguments</div>
        Tune epochs, batches, and rates.
    </a>
    <a href="#evaluator" class="box optional">
        <div class="header">Evaluator</div>
        Measure before and after.
    </a>
    <a href="#trainer" class="box">
        <div class="header">Trainer</div>
        Run the fine-tuning loop.
    </a>
</div>

## Model

Load a trainable PyTorch model:

```python
from relational_transformers import RelationalTransformer

model = RelationalTransformer("RelativeDB/rt-j-fp16", backend="torch")
```

Begin full fine-tuning from FP16 weights. Published reduced-precision checkpoints are
intended for evaluated inference deployments, not as substitutes for the training base.
The trainer rejects every backend except torch.

## Dataset

Each example contains a complete context and its label. Contexts may have different
lengths; the trainer pads and collates each mini-batch. Every context must share the same
embedding width and checkpoint contract.

```python
from relational_transformers import RelationalDataset, RelationalExample

train_dataset = RelationalDataset([
    RelationalExample(input=customer_batch, label=1.0)
    for customer_batch, label in training_pairs
])
```

### Dataset Format

Use a `RelationalBatch` per example to preserve typed number, text, datetime, and boolean
channels plus node, column, table, and foreign-key relations. The
[Dataset Overview](dataset_overview.md) covers accepted input types, label shapes, and
how to split relational data without leakage.

## Loss Function

Choose `binary`, `multiclass`, `multilabel`, `regression`, or `forecasting` through the
`problem_type` argument; the trainer instantiates the matching loss. When you skip it,
the trainer derives a default from the loaded checkpoint: regression tasks get
`RegressionLoss` and everything else gets `BinaryClassificationLoss`. See the
[Loss Overview](loss_overview.md) for shapes and custom losses.

## Training Arguments

`RelationalTrainingArguments` configures the loop. Every field has a working default:

| Argument | Default | Purpose |
| --- | --- | --- |
| `output_dir` | `"relational_model"` | where checkpoints are written |
| `num_train_epochs` | `1` | passes over the shuffled dataset |
| `per_device_train_batch_size` | `8` | examples per optimizer input |
| `learning_rate` | `1e-5` | AdamW learning rate |
| `weight_decay` | `1e-2` | AdamW weight decay |
| `max_grad_norm` | `1.0` | gradient clipping threshold |
| `gradient_accumulation_steps` | `1` | mini-batches per optimizer step |
| `seed` | `42` | shuffle and torch seeding |
| `logging_steps` | `10` | reserved for loss logging cadence |
| `save_strategy` | `"epoch"` | `"epoch"` saves after each epoch; anything else skips saving |

Gradient accumulation buys effective batch sizes larger than device memory: with
`per_device_train_batch_size=8` and `gradient_accumulation_steps=4`, each optimizer step
sees 32 examples.

## Evaluator

Run an evaluator before and after training so you have a measured delta:

| Evaluator | Task | Metrics |
| --- | --- | --- |
| `BinaryClassificationEvaluator` | binary | accuracy, precision, recall, F1 at a threshold |
| `RegressionEvaluator` | regression, forecasting | MAE, RMSE, R² |
| `AblationEvaluator` | any | mean and mean absolute prediction delta per named ablation |
| `SequentialEvaluator` | composition | merged metrics from several evaluators |

Evaluators are callables that take the model and return a metric dictionary.
`SequentialEvaluator` raises on duplicate metric names, so combined results stay
unambiguous.

## Trainer

`RelationalTrainer` fine-tunes every RT-J parameter and writes a regular Hugging
Face-style safetensors checkpoint. Each epoch shuffles the dataset, collates mini-batches,
computes the task loss on `target_scores`, clips gradients, and steps AdamW.
`trainer.train()` returns the final loss and step count, and `model.fit_head()` remains
the lighter path that encodes examples once and updates only a named task head.

## End-to-End Example

```{eval-rst}
.. sidebar:: Documentation

   #. :class:`~relational_transformers.RelationalTransformer`
   #. :class:`~relational_transformers.BinaryClassificationEvaluator`
   #. :class:`~relational_transformers.RelationalTrainingArguments`
   #. :class:`~relational_transformers.RelationalTrainer`
   #. :meth:`RelationalTrainer.train() <relational_transformers.RelationalTrainer.train>`

   **Related links:**

   - `Training Examples <training/examples.html>`_
   - `Full Fine-tuning <training/full_finetuning.html>`_
```

```python
from relational_transformers import (
    BinaryClassificationEvaluator,
    RelationalTrainer,
    RelationalTrainingArguments,
    RelationalTransformer,
)

# 1. Load the trainable model
model = RelationalTransformer("RelativeDB/rt-j-fp16")

# 2. Measure the pretrained baseline
evaluator = BinaryClassificationEvaluator(validation_examples)
before = evaluator(model)

# 3. Configure and run fine-tuning
trainer = RelationalTrainer(
    model=model,
    args=RelationalTrainingArguments(
        output_dir="models/churn",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
    ),
    train_dataset=train_examples,
    problem_type="binary",
)
result = trainer.train()

# 4. Measure again and keep both numbers with the artifact
after = evaluator(model)
print(before["f1"], "->", after["f1"], "in", result["steps"], "steps")
```

The checkpoint in `models/churn` reloads with `RelationalTransformer("models/churn")`.
PyTorch can serve it directly; ONNX requires export, and Triton requires a supported
classification checkpoint and CUDA runtime.

For GPU training, set `training_backend="triton"` in the arguments. TorchInductor then
compiles the forward and backward graphs to Triton CUDA kernels; the saved checkpoint
remains portable.

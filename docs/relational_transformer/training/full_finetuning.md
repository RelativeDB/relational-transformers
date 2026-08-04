# Full-model fine-tuning

`RelationalTrainer` updates every parameter of the loaded RT-J model: value encoders,
relational attention blocks, normalization scales, and decoder heads.

```python
from relational_transformers import (
    RelationalTrainer,
    RelationalTrainingArguments,
    RelationalTransformer,
)

model = RelationalTransformer("RelativeDB/rt-j-fp16")

args = RelationalTrainingArguments(
    output_dir="models/customer-churn",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=1e-5,
)
trainer = RelationalTrainer(model=model, args=args, train_dataset=examples)
result = trainer.train()
print(result)
# => {'train_loss': 0.31..., 'steps': 42}
```

Full tuning requires the torch backend; the trainer raises `RuntimeError` for any other.
The loss runs on the model's `target_scores` output, so this path fits scalar binary and
regression objectives. Multiclass and multilabel tasks go through
[head tuning](head_tuning.md).

## What a Training Step Does

Each epoch shuffles the examples with the configured seed, collates
`per_device_train_batch_size` contexts into one padded batch on the model's device, and
computes the task loss between `target_scores` and the labels. Gradients accumulate for
`gradient_accumulation_steps` mini-batches, get clipped to `max_grad_norm`, and AdamW
steps with the configured learning rate and weight decay. `trainer.losses` keeps every
mini-batch loss for inspection.

With `save_strategy="epoch"` the trainer writes a checkpoint to `output_dir` after each
epoch. Save directories keep the same `config.json` plus safetensors layout used by
Hugging Face checkpoints, so the result reloads through the normal constructor:

```python
tuned = RelationalTransformer("models/customer-churn")
```

## Triton-compiled GPU training

Set `training_backend="triton"` to compile the PyTorch forward and backward graphs with
TorchInductor's Triton CUDA code generation. This is the trainable Triton path; the
hand-tuned `backend="triton"` constructor remains an inference-only serving backend.

```python
args = RelationalTrainingArguments(
    output_dir="models/customer-churn",
    training_backend="triton",
    num_train_epochs=3,
    per_device_train_batch_size=8,
)
model = RelationalTransformer(device="cuda")
trainer = RelationalTrainer(model=model, args=args, train_dataset=train_examples)
trainer.train()
```

Triton training requires CUDA. Checkpoint saving still writes the ordinary portable
state dictionary, so the result reloads without a compiler dependency.

## Practical Notes

The default `problem_type` follows the loaded checkpoint: regression checkpoints train
with `RegressionLoss` and everything else with `BinaryClassificationLoss`. Pass
`problem_type=` explicitly when the label semantics differ from the checkpoint default.

Start from FP16 weights. Reduced-precision releases are inference artifacts, not the
fine-tuning base. Evaluate before and after with the same evaluator, as shown in the
[Training Overview](../training_overview.md#end-to-end-example).

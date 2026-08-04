# Full-model fine-tuning

```python
args = RelationalTrainingArguments(
    output_dir="customer-churn",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=1e-5,
)
trainer = RelationalTrainer(model=model, args=args, train_dataset=examples)
trainer.train()
```

Full tuning currently uses the PyTorch backend. Save directories retain the
same `config.json` plus safetensors layout used by Hugging Face checkpoints.

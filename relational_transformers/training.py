"""Frozen-head fitting and full-model fine-tuning."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .losses import loss_for


@dataclass
class RelationalExample:
    """One labeled context used for training or evaluation.

    ``target`` is required when ``input`` is a raw cell-vector array. Typed
    :class:`~relational_transformers.RelationalBatch` inputs already carry
    their target mask and leave it unset.
    """

    input: Any
    label: Any
    target: int | Sequence[int] | None = None


@dataclass
class RelationalTrainingArguments:
    output_dir: str = "relational_model"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    seed: int = 42
    logging_steps: int = 10
    save_strategy: str = "epoch"
    training_backend: str = "torch"


class TaskHead(nn.Module):
    def __init__(self, d_model: int, num_labels: int = 1, problem_type: str = "binary"):
        super().__init__()
        self.d_model = d_model
        self.num_labels = num_labels
        self.problem_type = problem_type
        self.projection = nn.Linear(d_model, num_labels)

    def forward(self, features: Tensor) -> Tensor:
        return self.projection(features)

    def save_pretrained(self, directory: str | Path) -> None:
        from safetensors.torch import save_file

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(), str(directory / "head.safetensors"))
        config = {
            "d_model": self.d_model,
            "num_labels": self.num_labels,
            "problem_type": self.problem_type,
        }
        (directory / "head_config.json").write_text(json.dumps(config, indent=2) + "\n")

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> TaskHead:
        from safetensors.torch import load_file

        directory = Path(directory)
        config = json.loads((directory / "head_config.json").read_text())
        head = cls(**config)
        head.load_state_dict(load_file(str(directory / "head.safetensors")))
        return head


def _loss(logits: Tensor, labels: Tensor, problem_type: str) -> Tensor:
    return loss_for(problem_type)(logits, labels)


def fit_head(
    transformer,
    examples: Sequence[RelationalExample],
    *,
    task: str,
    num_labels: int = 1,
    problem_type: str = "binary",
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> TaskHead:
    """Encode each example once and fit a lightweight task head."""

    if transformer.backend_name != "torch":
        raise RuntimeError("head fitting requires the torch backend")
    if not examples:
        raise ValueError("head fitting requires examples")
    features = []
    labels = []
    for example in examples:
        feature = transformer.encode(
            example.input,
            target=example.target,
            output_value="target_features",
            convert_to_numpy=False,
        )
        features.append(feature.squeeze(0).detach().cpu())
        labels.append(example.label)
    x = torch.stack(features).to(transformer.device)
    y = torch.as_tensor(labels, device=transformer.device)
    head = TaskHead(transformer.model.d_model, num_labels, problem_type).to(transformer.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    head.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(head(x), y, problem_type)
        loss.backward()
        optimizer.step()
    head.eval()
    transformer.heads[task] = head
    return head


class RelationalTrainer:
    """Small, dependency-free trainer for complete RT-J fine-tuning."""

    def __init__(
        self,
        *,
        model,
        args: RelationalTrainingArguments,
        train_dataset: Sequence[RelationalExample],
        task: str | None = None,
        problem_type: str | None = None,
    ) -> None:
        if model.backend_name != "torch":
            raise RuntimeError("full fine-tuning requires the torch backend")
        self.transformer = model
        self.args = args
        self.train_dataset = list(train_dataset)
        self.task = task
        self.problem_type = problem_type or (
            "regression" if model.task in ("regression", "forecasting", "reg") else "binary"
        )
        self.losses: list[float] = []

    def train(self) -> dict[str, float]:
        from .model import _collate_batches

        if not self.train_dataset:
            raise ValueError("train_dataset cannot be empty")
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        model = self.transformer.model
        if self.args.training_backend not in ("torch", "triton"):
            raise ValueError("training_backend must be 'torch' or 'triton'")
        if self.args.training_backend == "triton" and self.transformer.device.type != "cuda":
            raise RuntimeError("Triton training requires a CUDA device")
        model.train()
        train_model = model
        if self.args.training_backend == "triton":
            train_model = torch.compile(model, backend="inductor", dynamic=True)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay
        )
        batch_size = self.args.per_device_train_batch_size
        accumulation = self.args.gradient_accumulation_steps
        if accumulation < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        for _ in range(self.args.num_train_epochs):
            order = list(self.train_dataset)
            random.shuffle(order)
            for start in range(0, len(order), batch_size):
                examples = order[start : start + batch_size]
                batches = [
                    self.transformer._batch(example.input, target=example.target)
                    for example in examples
                ]
                batch = _collate_batches(batches).to(self.transformer.device)
                labels = torch.as_tensor(
                    [example.label for example in examples], device=self.transformer.device
                )
                logits = train_model(batch, output="target_scores").scores
                loss = _loss(logits, labels, self.problem_type)
                (loss / accumulation).backward()
                pending += 1
                is_last = start + batch_size >= len(order)
                if pending == accumulation or is_last:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    pending = 0
                self.losses.append(float(loss.detach()))
            if self.args.save_strategy == "epoch":
                self.save_model(self.args.output_dir)
        model.eval()
        return {"train_loss": self.losses[-1], "steps": len(self.losses)}

    def save_model(self, directory: str | Path | None = None) -> None:
        self.transformer.save_pretrained(directory or self.args.output_dir)

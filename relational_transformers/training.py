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


class HeadError(RuntimeError):
    """A feature-head artifact is missing, malformed, or misused."""


FEATURE_HEAD_TASKS = ("binary", "regression", "multiclass", "ranking")


class FineTunedHead:
    """A trained task head over the frozen backbone's target-cell features.

    The transformer is never updated; this is the small linear adapter that
    replaces a released checkpoint's zero-shot head. ``predict`` is plain
    numpy, so a fitted head serves anywhere. ``column_stats`` rides along
    opaquely (any object with ``to_dict()``, persisted as JSON and returned
    as a dict on load) so the artifact carries the preprocessing it was
    fitted under without this package interpreting it.
    """

    def __init__(self, weight, bias, *, task: str,
                 initial_loss=None, final_loss=None, seconds=None,
                 n_examples=None, classes: Sequence[Any] = (),
                 feat_mu=None, feat_sd=None,
                 column_stats=None, normalization_mode: str = "zero_shot"):
        import numpy as np

        self.weight = np.asarray(weight, np.float32)
        self.bias = np.asarray(bias, np.float32)
        self.task = task
        self.initial_loss = initial_loss
        self.final_loss = final_loss
        self.seconds = seconds
        self.n_examples = n_examples
        self.classes = tuple(classes)
        # Standardization statistics of the fitted features, so predict()
        # applies exactly the transform fit() saw.
        self.feat_mu = None if feat_mu is None else np.asarray(feat_mu, np.float32)
        self.feat_sd = None if feat_sd is None else np.asarray(feat_sd, np.float32)
        self.column_stats = column_stats
        self.normalization_mode = str(getattr(normalization_mode, "value",
                                              normalization_mode))

    @property
    def n_outputs(self) -> int:
        return int(self.weight.shape[0])

    @property
    def d_model(self) -> int:
        return int(self.weight.shape[1])

    def _sidecar(path: str) -> str:                      # noqa: N805
        return str(path) + ".preproc.json"

    def save(self, path) -> str:
        """Persist the head plus the preprocessing it was fitted under."""
        from safetensors.numpy import save_file

        save_file({"weight": self.weight, "bias": self.bias}, str(path))
        stats = self.column_stats
        if stats is not None and hasattr(stats, "to_dict"):
            stats = stats.to_dict()
        side = {
            "task": self.task,
            "feat_mu": None if self.feat_mu is None else self.feat_mu.tolist(),
            "feat_sd": None if self.feat_sd is None else self.feat_sd.tolist(),
            "column_stats": stats,
            "normalization_mode": self.normalization_mode,
            "classes": [str(c) for c in self.classes],
        }
        Path(FineTunedHead._sidecar(path)).write_text(json.dumps(side))
        return str(path)

    @staticmethod
    def load(path) -> "FineTunedHead":
        import numpy as np
        from safetensors.numpy import load_file

        try:
            tensors = load_file(str(path))
            weight, bias = tensors["weight"], tensors["bias"]
        except Exception as e:
            raise HeadError(
                f"loading a fine-tuned head from {path!r} failed: {e}") from e
        side_path = Path(FineTunedHead._sidecar(path))
        if not side_path.exists():
            raise HeadError(
                f"{side_path} is missing: this head was saved without its "
                f"preprocessing, and serving it would apply the wrong scale "
                f"to every numeric cell. Refit rather than loading it.")
        side = json.loads(side_path.read_text())
        return FineTunedHead(
            weight, bias, task=str(side.get("task", "binary")),
            classes=tuple(side.get("classes") or ()),
            feat_mu=(None if side.get("feat_mu") is None
                     else np.asarray(side["feat_mu"], np.float32)),
            feat_sd=(None if side.get("feat_sd") is None
                     else np.asarray(side["feat_sd"], np.float32)),
            column_stats=side.get("column_stats"),
            normalization_mode=side.get("normalization_mode", "reference"))

    def predict(self, features):
        """Score frozen features ``[N, d_model]`` -> logits ``[N, n_outputs]``."""
        import numpy as np

        f = np.asarray(features, np.float32)
        if self.feat_mu is not None:
            f = (f - self.feat_mu) / self.feat_sd
        f = np.ascontiguousarray(f, np.float32)
        if f.ndim != 2 or f.shape[1] != self.d_model:
            raise HeadError(
                f"features must be [N, {self.d_model}], got {f.shape}")
        return f @ self.weight.T + self.bias

    def __repr__(self) -> str:
        loss = ""
        if self.initial_loss is not None and self.final_loss is not None:
            loss = f" loss {self.initial_loss:.4f}->{self.final_loss:.4f}"
        n = f" on {self.n_examples} examples" if self.n_examples else ""
        return f"<FineTunedHead {self.task}{n}{loss}>"


def _feature_loss_fn(task: str, group_offsets, n_groups: int):
    from .losses import ListwiseRankingLoss, loss_for

    if task == "ranking":
        ranking = ListwiseRankingLoss()
        offsets = group_offsets[:n_groups + 1]
        return lambda logits, labels: ranking(logits, labels, offsets)
    loss = loss_for(task)
    if task in ("binary", "regression"):
        return lambda logits, labels: loss(logits.reshape(-1), labels)
    return loss


def fit_feature_head(features, labels, task: str, *,
                     classes: Sequence[Any] = (),
                     group_offsets=None, n_groups: int = 0,
                     epochs: int = 100, learning_rate: float = 1e-3,
                     weight_decay: float = 1e-4,
                     class_embeddings=None, text_decoder=None,
                     column_stats=None,
                     normalization_mode: str = "zero_shot") -> FineTunedHead:
    """Fit a task head on frozen features ``[N, d_model]`` with AdamW.

    ``task`` is one of ``binary``, ``regression``, ``multiclass``, or
    ``ranking`` (grouped by ``group_offsets``). Features are standardized per
    dimension before fitting: the backbone's target-cell features sit in a
    very narrow cone (mean pairwise cosine 0.9976 on a 240-issue sample), and
    without standardization a linear head fits only its bias.

    For multiclass, pass ``class_embeddings`` (normalized label embeddings,
    ``[C, d_text]``) together with the checkpoint's ``text_decoder`` (the
    ``d_model -> d_text`` linear, e.g. ``model.model.dec_dict["text"]``) to
    seed the head in the checkpoint's own class-embedding basis, so training
    starts from the zero-shot ordering.
    """
    import numpy as np

    if task not in FEATURE_HEAD_TASKS:
        raise ValueError(f"task must be one of {FEATURE_HEAD_TASKS}")
    n_outputs = len(classes) if task == "multiclass" else 1
    feats = np.asarray(features, np.float32).reshape(len(labels), -1)
    d_model = feats.shape[1]
    feat_mu = np.ascontiguousarray(feats.mean(0), np.float32)
    feat_sd = np.ascontiguousarray(feats.std(0) + 1e-6, np.float32)
    feats = (feats - feat_mu) / feat_sd

    head = TaskHead(d_model, n_outputs, problem_type=task)
    projection = head.projection
    if task == "multiclass" and class_embeddings is not None and text_decoder is not None:
        class_emb = torch.as_tensor(
            np.ascontiguousarray(class_embeddings, np.float32))
        decoder = text_decoder.to("cpu").float()
        weight_raw = class_emb @ decoder.weight            # [C, d_model]
        bias_raw = class_emb @ decoder.bias                # [C]
        mu = torch.as_tensor(feat_mu)
        sd = torch.as_tensor(feat_sd)
        with torch.no_grad():
            projection.weight.copy_(weight_raw * sd)
            projection.bias.copy_(bias_raw + weight_raw @ mu)

    import time

    x = torch.as_tensor(feats)
    y = torch.as_tensor(np.ascontiguousarray(labels, np.float32))
    offsets = (np.ascontiguousarray(group_offsets, np.int64)
               if group_offsets is not None else np.zeros(1, np.int64))
    loss_fn = _feature_loss_fn(task, offsets, n_groups)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    started = time.perf_counter()
    initial_loss = None
    loss_value = None
    head.train()
    for _ in range(max(int(epochs), 1)):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(head(x), y)
        if initial_loss is None:
            initial_loss = float(loss.detach())
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
    head.eval()

    return FineTunedHead(
        projection.weight.detach().numpy(),
        projection.bias.detach().numpy(),
        task=task,
        initial_loss=initial_loss,
        final_loss=loss_value,
        seconds=time.perf_counter() - started,
        n_examples=int(y.shape[0]), classes=classes,
        feat_mu=feat_mu, feat_sd=feat_sd,
        column_stats=column_stats,
        normalization_mode=normalization_mode)

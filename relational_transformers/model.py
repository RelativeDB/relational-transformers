"""Sentence-Transformers-style high-level API for RT-J."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .batch import RelationalBatch
from .checkpoints import (
    load_state,
    model_dimensions,
    resolve_checkpoint,
    resolve_config,
    save_checkpoint,
)
from .constants import BATCH_FIELDS
from .onnx import OnnxBackend, export_onnx
from .torch_model import ModelOutput, RTJModel


class RelationalTransformer:
    """Load and run an RT-J checkpoint.

    Args:
        model_name_or_path: Hugging Face repository, local checkpoint directory,
            or ONNX file.
        task: ``"classification"`` (default) or ``"regression"``.
        backend: ``"torch"``, ``"triton"``, ``"onnx"``, or ``"meta"``.
        device: PyTorch device. Auto-selects CUDA, MPS, then CPU.
    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        task: str = "classification",
        backend: str = "torch",
        device: str | torch.device | None = None,
        revision: str | None = None,
        compile: bool = False,
        providers=None,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.task = task
        self.backend_name = backend
        self.device = torch.device(device or self._default_device())
        self.config: dict = {}
        self.model: RTJModel | None = None
        self.backend: Any = None
        self.heads: dict[str, torch.nn.Module] = {}

        if backend == "onnx":
            self.backend = OnnxBackend(model_name_or_path, providers=providers)
            return
        if backend == "meta":
            self.config = resolve_config(model_name_or_path, task=task, revision=revision)
            self.model = RTJModel(**model_dimensions(self.config), device="meta")
            self.backend = self.model
            return

        self.config, checkpoint = resolve_checkpoint(
            model_name_or_path, task=task, revision=revision
        )
        if backend == "triton":
            if task not in ("classification", "binary", "ranking", "clf"):
                raise ValueError("the Triton backend currently serves classification target scores")
            from .backends.triton_model import RTJTriton

            self.backend = RTJTriton(checkpoint)
            return
        if backend != "torch":
            raise ValueError("backend must be 'torch', 'triton', 'onnx', or 'meta'")
        state = load_state(checkpoint)
        legacy = not any(key.endswith(".wg.weight") for key in state)
        self.model = RTJModel(**model_dimensions(self.config), legacy_attn=legacy)
        self.model.load_state_dict(state)
        if self.device.type == "cpu":
            self.model = self.model.float()
        self.model = self.model.to(self.device).eval()
        if compile:
            self.model = torch.compile(self.model, dynamic=True)
        self.backend = self.model

    @staticmethod
    def _default_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _batch(self, inputs, *, target: int | Sequence[int] | None = None) -> RelationalBatch:
        if isinstance(inputs, RelationalBatch):
            return inputs
        if isinstance(inputs, Mapping):
            return RelationalBatch.from_mapping(inputs)
        if isinstance(inputs, (np.ndarray, torch.Tensor)):
            if target is None:
                raise ValueError("target is required when passing cell vectors directly")
            return RelationalBatch.from_text_cells(inputs, target=target)
        if isinstance(inputs, Sequence) and inputs:
            batches = [self._batch(item, target=target) for item in inputs]
            return _collate_batches(batches)
        raise TypeError(
            "inputs must be cell vectors, RelationalBatch, mapping, or a non-empty sequence"
        )

    def forward(self, inputs, *, output="target_scores", target=None) -> ModelOutput:
        batch = self._batch(inputs, target=target)
        if self.backend_name == "meta":
            raise RuntimeError("meta models inspect structure but cannot run inference")
        if self.backend_name == "onnx":
            if output != "target_scores":
                raise ValueError("ONNX runtime currently exposes target_scores")
            return ModelOutput(scores=torch.from_numpy(self.backend.predict(batch)))
        if self.backend_name == "triton":
            if output != "target_scores":
                raise ValueError("Triton currently exposes target_scores")
            return ModelOutput(scores=torch.from_numpy(self.backend.predict(batch.numpy())))
        with torch.inference_mode():
            return self.model(batch.to(self.device), output=output)

    def predict(
        self,
        inputs,
        *,
        target=None,
        task_head: str | None = None,
        activation: str | None = None,
        convert_to_numpy=True,
    ):
        if task_head is not None:
            if task_head not in self.heads:
                raise KeyError(f"no fitted task head named {task_head!r}")
            features = self.encode(
                inputs, target=target, output_value="target_features", convert_to_numpy=False
            )
            with torch.inference_mode():
                scores = self.heads[task_head](features)
            problem_type = self.heads[task_head].problem_type
        else:
            scores = self.forward(inputs, output="target_scores", target=target).scores
            problem_type = self.task
        if activation is None:
            if problem_type in ("classification", "binary", "multilabel", "clf"):
                activation = "sigmoid"
            elif problem_type == "multiclass":
                activation = "softmax"
        if activation == "sigmoid":
            scores = scores.sigmoid()
        elif activation == "softmax":
            scores = scores.softmax(dim=-1)
        elif activation not in (None, "identity"):
            raise ValueError("activation must be None, 'identity', 'sigmoid', or 'softmax'")
        if convert_to_numpy:
            values = scores.detach().cpu().numpy()
            return float(values[0]) if values.shape == (1,) else values
        return scores

    def encode(self, inputs, *, target=None, output_value="embeddings", convert_to_numpy=True):
        output = "target_features" if output_value == "target_features" else "embeddings"
        result = self.forward(inputs, output=output, target=target)
        values = result.features if output == "target_features" else result.embeddings
        return values.detach().cpu().numpy() if convert_to_numpy else values

    def export_onnx(self, path, example, *, target=None, opset_version=18):
        if self.backend_name != "torch" or self.model is None:
            raise RuntimeError("ONNX export requires a loaded torch backend")
        batch = self._batch(example, target=target).to(self.device)
        return export_onnx(self.model, batch, path, opset_version=opset_version)

    def save_pretrained(self, directory):
        if self.model is None or self.backend_name != "torch":
            raise RuntimeError("save_pretrained requires a torch model")
        save_checkpoint(self.model, directory, self.config)

    def fit_head(self, examples, **kwargs):
        from .training import fit_head

        return fit_head(self, examples, **kwargs)

    def get_model_kwargs(self) -> dict:
        """Return dimensions without materializing weights (especially useful on meta)."""
        return model_dimensions(self.config)


def _collate_batches(batches: Sequence[RelationalBatch]) -> RelationalBatch:
    if any(batch.batch_size != 1 for batch in batches):
        raise ValueError("sequence collation expects one context per RelationalBatch")
    width = batches[0].d_text
    if any(batch.d_text != width for batch in batches):
        raise ValueError("all contexts must have the same d_text")
    longest = max(batch.sequence_length for batch in batches)
    output = {}
    for name in BATCH_FIELDS:
        rows = []
        for batch in batches:
            tensor = getattr(batch, name)
            missing = longest - batch.sequence_length
            if missing:
                shape = list(tensor.shape)
                shape[1] = missing
                if name == "f2p_nbr_idxs":
                    padding = torch.full(shape, -1, dtype=tensor.dtype)
                elif name == "is_padding":
                    padding = torch.ones(shape, dtype=tensor.dtype)
                else:
                    padding = torch.zeros(shape, dtype=tensor.dtype)
                tensor = torch.cat([tensor, padding], dim=1)
            rows.append(tensor)
        output[name] = torch.cat(rows, dim=0)
    return RelationalBatch(**output)

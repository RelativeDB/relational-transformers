"""ONNX export and ONNX Runtime inference."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from .batch import RelationalBatch
from .constants import BATCH_FIELDS


class _OnnxModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *values):
        batch = SimpleNamespace(**dict(zip(BATCH_FIELDS, values, strict=True)))
        return self.model(batch, output="target_scores").scores


def export_onnx(model, batch: RelationalBatch, path: str | Path, *, opset_version: int = 18):
    """Export target-score inference with dynamic batch and sequence axes."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    inputs = tuple(batch.as_dict()[name] for name in BATCH_FIELDS)
    dynamic = {name: {0: "batch", 1: "cells"} for name in BATCH_FIELDS}
    dynamic["target_scores"] = {0: "batch"}
    torch.onnx.export(
        _OnnxModule(model).eval(),
        inputs,
        str(path),
        input_names=list(BATCH_FIELDS),
        output_names=["target_scores"],
        dynamic_axes=dynamic,
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )
    return path


class OnnxBackend:
    def __init__(self, path: str | Path, providers=None):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("install relational-transformers[onnx] for ONNX inference") from exc
        self.session = ort.InferenceSession(str(path), providers=providers)

    def predict(self, batch: RelationalBatch) -> np.ndarray:
        values = {}
        for name, tensor in batch.as_dict().items():
            array = tensor.detach().cpu().numpy()
            if array.dtype == np.bool_:
                values[name] = array
            elif tensor.is_floating_point():
                values[name] = array.astype(np.float32)
            else:
                values[name] = array.astype(np.int64)
        return self.session.run(["target_scores"], values)[0]

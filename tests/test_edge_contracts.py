from __future__ import annotations

import json
import sys

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from relational_transformers import (
    AblationEvaluator,
    BinaryClassificationEvaluator,
    RelationalBatch,
    RelationalTrainer,
    RelationalTrainingArguments,
    RelationalTransformer,
)
from relational_transformers.checkpoints import load_state, model_dimensions, resolve_config
from relational_transformers.losses import loss_for
from relational_transformers.model import _collate_batches
from relational_transformers.quantization import main as quantization_main


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("node_idxs", np.zeros((1, 2, 1)), "node_idxs must have shape"),
        ("col_name_idxs", np.zeros((1, 1)), "col_name_idxs must have shape"),
        ("f2p_nbr_idxs", np.zeros((1, 2, 4)), "f2p_nbr_idxs must have shape"),
        ("number_values", np.zeros((1, 2, 2)), "number_values must have shape"),
        ("text_values", np.zeros((1, 2)), "text_values must have shape"),
        ("col_name_values", np.zeros((1, 2, 3)), "col_name_values must match"),
    ],
)
def test_batch_shape_errors_are_specific(customer_context_factory, field, value, message):
    values = customer_context_factory(order_amounts=(), support_summary=None).numpy()
    values[field] = value

    with pytest.raises(ValueError, match=message):
        RelationalBatch.from_mapping(values)


def test_batch_casts_integer_value_channels_and_optional_dtype():
    cells = np.ones((3, 8), np.int64)
    batch = RelationalBatch.from_text_cells(
        cells,
        target=[0, 2],
        parents={1: [100]},
        node_idxs=[100, 101, 102],
        table_idxs=[0, 1, 1],
    )

    converted = batch.to("cpu", dtype=torch.float64)

    assert batch.f2p_nbr_idxs[0, 1, 0] == 100
    assert converted.text_values.dtype == torch.float64
    assert converted.node_idxs.dtype == torch.int64


def test_checkpoint_file_resolution_pt_loading_and_legacy_norm_name(tmp_path):
    config = {
        "num_blocks": 1,
        "d_model": 4,
        "d_text": 2,
        "num_heads": 1,
        "d_ff": 8,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"model": {"norm_out.weight": torch.ones(4)}}, checkpoint)

    assert resolve_config(checkpoint) == config
    state = load_state(checkpoint)

    assert "norm_out.scale" in state
    with pytest.raises(ValueError, match="missing"):
        model_dimensions({"d_model": 4})


def test_q4_rejects_non_byte_payload_and_ignores_orphan_scale(tmp_path):
    path = tmp_path / "broken.safetensors"
    save_file(
        {
            "projection.weight": torch.zeros((1, 16), dtype=torch.int8),
            "projection.weight.q4_scale": torch.ones((1, 2), dtype=torch.float16),
            "orphan.q4_scale": torch.ones((1, 2), dtype=torch.float16),
        },
        path,
    )

    with pytest.raises(ValueError, match="invalid Q4 payload"):
        load_state(path)


def test_high_level_backend_and_output_errors(tiny_checkpoint, customer_context_factory):
    context = customer_context_factory()
    with pytest.raises(ValueError, match="backend must be"):
        RelationalTransformer(tiny_checkpoint, backend="unknown")
    with pytest.raises(ValueError, match="Triton backend"):
        RelationalTransformer(tiny_checkpoint, backend="triton", task="regression")

    model = RelationalTransformer(tiny_checkpoint, device="cpu")
    with pytest.raises(KeyError, match="no fitted task head"):
        model.predict(context, task_head="missing")
    with pytest.raises(ValueError, match="activation must be"):
        model.predict(context, activation="tanh")
    tensor = model.predict(context, convert_to_numpy=False)
    assert isinstance(tensor, torch.Tensor)

    meta = RelationalTransformer(tiny_checkpoint, backend="meta")
    with pytest.raises(RuntimeError, match="ONNX export requires"):
        meta.export_onnx("unused.onnx", context)
    with pytest.raises(RuntimeError, match="save_pretrained requires"):
        meta.save_pretrained("unused")


def test_default_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert RelationalTransformer._default_device() == "cpu"


def test_collation_rejects_already_batched_context(customer_context_factory):
    context = customer_context_factory()
    batched = _collate_batches([context, context])

    with pytest.raises(ValueError, match="one context"):
        _collate_batches([batched])


def test_evaluator_and_loss_require_supported_nonempty_inputs(tiny_checkpoint):
    model = RelationalTransformer(tiny_checkpoint, device="cpu")
    with pytest.raises(ValueError, match="at least one example"):
        BinaryClassificationEvaluator([])(model)
    with pytest.raises(ValueError, match="named ablation"):
        AblationEvaluator([object()], {})(model)
    with pytest.raises(ValueError, match="unsupported problem_type"):
        loss_for("survival")


def test_training_requires_torch_backend_and_nonempty_data(tiny_checkpoint):
    meta = RelationalTransformer(tiny_checkpoint, backend="meta")
    with pytest.raises(RuntimeError, match="requires the torch backend"):
        RelationalTrainer(
            model=meta,
            args=RelationalTrainingArguments(),
            train_dataset=[],
        )

    model = RelationalTransformer(tiny_checkpoint, device="cpu")
    trainer = RelationalTrainer(
        model=model,
        args=RelationalTrainingArguments(),
        train_dataset=[],
    )
    with pytest.raises(ValueError, match="cannot be empty"):
        trainer.train()
    with pytest.raises(ValueError, match="requires examples"):
        model.fit_head([], task="empty")


def test_quantization_cli_writes_selected_task(tiny_checkpoint, tmp_path, monkeypatch):
    output = tmp_path / "fp8-cli"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "relational-transformers-quantize",
            str(tiny_checkpoint),
            str(output),
            "--task",
            "classification",
        ],
    )

    assert quantization_main() == 0
    assert (output / "classification" / "model.fp8.safetensors").is_file()
    assert not (output / "regression").exists()

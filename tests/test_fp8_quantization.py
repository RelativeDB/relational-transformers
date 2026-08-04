from __future__ import annotations

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from relational_transformers import RelationalTransformer
from relational_transformers.checkpoints import load_state
from relational_transformers.quantization import (
    quantize_checkpoint_fp8,
    quantize_model_fp8,
    quantize_tensor_fp8,
)


def test_fp8_quantizes_matrix_weights_but_not_biases_or_norms():
    matrix = torch.linspace(-1, 1, 32).reshape(4, 8)
    bias = torch.linspace(-1, 1, 4)

    quantized_matrix = quantize_tensor_fp8("projection.weight", matrix)
    retained_bias = quantize_tensor_fp8("projection.bias", bias)

    assert quantized_matrix.dtype == torch.float8_e4m3fn
    assert retained_bias.dtype == torch.float32
    assert retained_bias is bias


def test_fp8_checkpoint_records_metadata_and_loads_portably(tmp_path):
    source = tmp_path / "source.safetensors"
    destination = tmp_path / "model.fp8.safetensors"
    weight = torch.randn(8, 8)
    save_file({"projection.weight": weight, "projection.bias": torch.ones(8)}, source)

    quantize_checkpoint_fp8(source, destination)

    with safe_open(destination, framework="pt", device="cpu") as handle:
        assert handle.metadata()["quantization"] == "fp8_e4m3fn"
        assert handle.get_tensor("projection.weight").dtype == torch.float8_e4m3fn
    loaded = load_state(destination)
    assert loaded["projection.weight"].dtype == torch.float32
    torch.testing.assert_close(loaded["projection.weight"], weight, atol=0.07, rtol=0.07)


def test_quantized_model_round_trips_all_tasks_and_predicts(
    tiny_checkpoint, customer_context_factory, tmp_path
):
    output = quantize_model_fp8(tiny_checkpoint, tmp_path / "fp8")
    context = customer_context_factory()

    for task in ("classification", "regression"):
        config = json.loads((output / task / "config.json").read_text())
        assert config["checkpoint_file"] == "model.fp8.safetensors"
        assert config["quantization"]["format"] == "fp8_e4m3fn"
        stored = load_file(output / task / "model.fp8.safetensors")
        assert stored["blocks.0.attns.col.wq.weight"].dtype == torch.float8_e4m3fn
        model = RelationalTransformer(output, task=task, device="cpu")
        assert isinstance(model.predict(context), float)


def test_fp8_prediction_stays_close_to_source_model(
    tiny_checkpoint, customer_context_factory, tmp_path
):
    context = customer_context_factory()
    source = RelationalTransformer(tiny_checkpoint, device="cpu")
    output = quantize_model_fp8(
        tiny_checkpoint,
        tmp_path / "fp8",
        tasks=("classification",),
    )
    quantized = RelationalTransformer(output, device="cpu")

    expected = source.predict(context, activation="identity")
    actual = quantized.predict(context, activation="identity")

    assert actual == pytest.approx(expected, abs=0.15)


def test_fp8_requires_safetensors_input(tmp_path):
    with pytest.raises(ValueError, match="requires a safetensors"):
        quantize_checkpoint_fp8(tmp_path / "model.pt", tmp_path / "model.fp8.safetensors")

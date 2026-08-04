from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from relational_transformers import RelationalTransformer

pytestmark = [
    pytest.mark.hub,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("RUN_HUB_TESTS") != "1",
        reason="set RUN_HUB_TESTS=1 to download and verify published checkpoints",
    ),
]


@pytest.mark.parametrize(
    "model_id",
    [
        "RelativeDB/rt-j-fp16",
        "RelativeDB/rt-j-fp8",
        "RelativeDB/rt-j-int8",
        "RelativeDB/rt-j-int4",
    ],
)
def test_published_classifiers_load_and_score_realistic_context(model_id, customer_context_factory):
    context = customer_context_factory(d_text=384)
    model = RelationalTransformer(model_id, device="cpu")

    logit = model.predict(context, activation="identity")
    probability = model.predict(context)

    assert math.isfinite(logit)
    assert math.isfinite(probability)
    assert 0.0 < probability < 1.0


def test_published_regression_checkpoint_loads_and_scores(customer_context_factory):
    context = customer_context_factory(d_text=384)
    model = RelationalTransformer("RelativeDB/rt-j-fp16", task="regression", device="cpu")

    prediction = model.predict(context)

    assert math.isfinite(prediction)


def test_published_model_responds_to_explicit_support_ablation(customer_context_factory):
    context = customer_context_factory(d_text=384)
    without_support = context.ablate([11, 12])
    model = RelationalTransformer("RelativeDB/rt-j-fp16", device="cpu")

    full, ablated = model.predict(
        [context, without_support],
        activation="identity",
    )

    assert abs(float(full - ablated)) > 1e-6


def test_published_meta_model_downloads_only_configuration():
    model = RelationalTransformer(backend="meta")

    assert model.get_model_kwargs() == {
        "num_blocks": 12,
        "d_model": 512,
        "d_text": 384,
        "num_heads": 8,
        "d_ff": 2048,
    }
    assert all(parameter.device.type == "meta" for parameter in model.model.parameters())


def test_published_checkpoint_exports_and_runs_with_onnx_runtime(
    customer_context_factory, tmp_path
):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    context = customer_context_factory(d_text=384)
    larger_context = customer_context_factory(d_text=384, order_amounts=(129.0, 58.5, 240.0))
    torch_model = RelationalTransformer(device="cpu")
    path = torch_model.export_onnx(Path(tmp_path) / "rt-j-classification.onnx", context)
    onnx_model = RelationalTransformer(path, backend="onnx")

    expected = torch_model.predict([context, larger_context], activation="identity")
    actual = onnx_model.predict([context, larger_context], activation="identity")

    assert path.stat().st_size > 100 * 1024 * 1024
    assert all(math.isfinite(float(score)) for score in actual)
    assert actual == pytest.approx(expected, rel=1e-4, abs=1e-4)

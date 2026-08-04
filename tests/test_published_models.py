from __future__ import annotations

import math
import os

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


def test_published_meta_model_downloads_only_configuration():
    model = RelationalTransformer("RelativeDB/rt-j-fp16", backend="meta")

    assert model.get_model_kwargs() == {
        "num_blocks": 12,
        "d_model": 512,
        "d_text": 384,
        "num_heads": 8,
        "d_ff": 2048,
    }
    assert all(parameter.device.type == "meta" for parameter in model.model.parameters())

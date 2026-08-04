from __future__ import annotations

import os

import pytest
import torch

from relational_transformers import RelationalTransformer

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.hub,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("RUN_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
        reason="set RUN_CUDA_TESTS=1 on a CUDA host to verify Triton kernels",
    ),
]


def test_triton_matches_pytorch_on_related_customer_context(customer_context_factory):
    pytest.importorskip("triton")
    context = customer_context_factory(d_text=384)
    torch_model = RelationalTransformer("RelativeDB/rt-j-fp16", backend="torch", device="cuda")
    triton_model = RelationalTransformer("RelativeDB/rt-j-fp16", backend="triton")

    expected = torch_model.predict(context, activation="identity")
    actual = triton_model.predict(context, activation="identity")

    assert actual == pytest.approx(expected, abs=5e-2)

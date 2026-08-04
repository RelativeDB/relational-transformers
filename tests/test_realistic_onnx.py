from __future__ import annotations

import numpy as np
import pytest
import torch

from relational_transformers import RelationalTransformer

pytestmark = pytest.mark.onnx


def test_onnx_handles_dynamic_batch_and_context_lengths(
    tiny_checkpoint, customer_context_factory, tmp_path
):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    torch_model = RelationalTransformer(tiny_checkpoint, device="cpu")
    export_example = customer_context_factory(order_amounts=(19.0,))
    path = torch_model.export_onnx(tmp_path / "customer-risk.onnx", export_example)
    onnx_model = RelationalTransformer(path, backend="onnx")

    contexts = [
        customer_context_factory(),
        customer_context_factory(
            customer_id=44_903_501,
            order_amounts=(11.0,),
            support_summary=None,
        ),
    ]
    expected = torch_model.predict(contexts, activation="identity")
    actual = onnx_model.predict(contexts, activation="identity")

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_onnx_accepts_bfloat16_cell_embeddings(
    tiny_checkpoint, customer_context_factory, tmp_path
):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    torch_model = RelationalTransformer(tiny_checkpoint, device="cpu")
    context = customer_context_factory()
    path = torch_model.export_onnx(tmp_path / "customer-risk.onnx", context)
    onnx_model = RelationalTransformer(path, backend="onnx")
    expected = torch_model.predict(context, activation="identity")
    context.text_values = context.text_values.to(torch.bfloat16)
    context.col_name_values = context.col_name_values.to(torch.bfloat16)

    actual = onnx_model.predict(context, activation="identity")

    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


def test_onnx_rejects_outputs_not_present_in_export(
    tiny_checkpoint, customer_context_factory, tmp_path
):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    torch_model = RelationalTransformer(tiny_checkpoint, device="cpu")
    context = customer_context_factory()
    path = torch_model.export_onnx(tmp_path / "customer-risk.onnx", context)
    onnx_model = RelationalTransformer(path, backend="onnx")

    with pytest.raises(ValueError, match="currently exposes target_scores"):
        onnx_model.forward(context, output="target_features")

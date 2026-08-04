import numpy as np
import pytest
import torch

from relational_transformers import (
    RelationalBatch,
    RelationalExample,
    RelationalTransformer,
    RTJModel,
    TaskHead,
)
from relational_transformers.checkpoints import save_checkpoint


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(5)
    model = RTJModel(num_blocks=1, d_model=8, d_text=4, num_heads=2, d_ff=16)
    config = {
        "task_type": "clf",
        "model": {"num_blocks": 1, "d_model": 8, "d_text": 4, "num_heads": 2, "d_ff": 16},
    }
    save_checkpoint(model, tmp_path, config)
    return tmp_path


def cells(seed=1):
    return np.random.default_rng(seed).normal(size=(4, 8)).astype("float32")


def test_hub_style_constructor_and_meta(checkpoint):
    model = RelationalTransformer(checkpoint, device="cpu")
    score = model.predict(cells(), target=0)
    assert isinstance(score, float)

    meta = RelationalTransformer(checkpoint, backend="meta")
    assert meta.get_model_kwargs()["d_model"] == 8
    assert next(meta.model.parameters()).device.type == "meta"


def test_head_fitting_uses_frozen_target_features(checkpoint):
    model = RelationalTransformer(checkpoint, device="cpu")
    examples = [
        RelationalExample(RelationalBatch.from_text_cells(cells(i), target=0), i % 2)
        for i in range(4)
    ]
    before = {name: value.clone() for name, value in model.model.state_dict().items()}
    head = model.fit_head(examples, task="binary", epochs=2)
    assert head.projection.out_features == 1
    assert all(torch.equal(before[name], value) for name, value in model.model.state_dict().items())


def test_multiclass_head_predicts_probabilities(checkpoint):
    model = RelationalTransformer(checkpoint, device="cpu")
    model.heads["segment"] = TaskHead(8, num_labels=3, problem_type="multiclass")
    probabilities = model.predict(cells(), target=0, task_head="segment")
    assert probabilities.shape == (1, 3)
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0, atol=1e-6)


def test_onnx_round_trip(checkpoint, tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    model = RelationalTransformer(checkpoint, device="cpu")
    batch = RelationalBatch.from_text_cells(cells(), target=0)
    expected = model.forward(batch).scores.numpy()
    path = model.export_onnx(tmp_path / "model.onnx", batch)
    onnx_model = RelationalTransformer(path, backend="onnx")
    actual = onnx_model.forward(batch).scores.numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    larger = RelationalBatch.from_text_cells(
        np.random.default_rng(9).normal(size=(7, 8)).astype("float32"), target=2
    )
    expected_larger = model.forward(larger).scores.numpy()
    actual_larger = onnx_model.forward(larger).scores.numpy()
    np.testing.assert_allclose(actual_larger, expected_larger, rtol=1e-5, atol=1e-6)

from __future__ import annotations

import os

import pytest
import torch

from relational_transformers import (
    RelationalExample,
    RelationalTrainer,
    RelationalTrainingArguments,
    RelationalTransformer,
)

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


def test_triton_regression_matches_pytorch(customer_context_factory):
    pytest.importorskip("triton")
    context = customer_context_factory(
        customer_id=77_001, order_amounts=(21.0, 89.0), d_text=384
    )
    torch_model = RelationalTransformer(
        "RelativeDB/rt-j-fp16", task="regression", device="cuda"
    )
    triton_model = RelationalTransformer(
        "RelativeDB/rt-j-fp16", task="regression", backend="triton"
    )

    expected = torch_model.predict(context, activation="identity")
    actual = triton_model.predict(context, activation="identity")

    assert actual == pytest.approx(expected, abs=5e-2)


def test_triton_exposes_every_serving_output(customer_context_factory):
    pytest.importorskip("triton")
    context = customer_context_factory(d_text=384)
    torch_model = RelationalTransformer(
        "RelativeDB/rt-j-fp16", backend="torch", device="cuda")
    triton_model = RelationalTransformer(
        "RelativeDB/rt-j-fp16", backend="triton")

    expected_tokens = torch_model.forward(
        context, output="token_scores").token_scores
    actual_tokens = triton_model.forward(
        context, output="token_scores").token_scores
    expected_features = torch_model.forward(
        context, output="target_features").features
    actual_features = triton_model.forward(
        context, output="target_features").features
    expected_both = torch_model.forward(
        context, output="target_scores_and_text")
    actual_both = triton_model.forward(
        context, output="target_scores_and_text")

    torch.testing.assert_close(actual_tokens, expected_tokens,
                               rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(actual_features, expected_features,
                               rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(actual_both.scores, expected_both.scores,
                               rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(actual_both.target_text,
                               expected_both.target_text,
                               rtol=5e-2, atol=5e-2)


def test_triton_returns_placeholder_for_phantom_batch_rows(customer_context_factory):
    pytest.importorskip("triton")
    real = customer_context_factory(d_text=384)
    phantom = customer_context_factory(d_text=384)
    phantom.is_padding[:] = True
    phantom.is_targets[:] = False
    model = RelationalTransformer("RelativeDB/rt-j-fp16", backend="triton")

    scores = model.predict([real, phantom], activation="identity")

    assert scores.shape == (2,)
    assert scores[1] == 0.0


def test_fp8_triton_matches_portable_backend(customer_context_factory):
    pytest.importorskip("triton")
    fp8_path = os.environ.get("FP8_MODEL_PATH")
    if not fp8_path:
        pytest.skip("set FP8_MODEL_PATH to validate a local FP8 release artifact")
    context = customer_context_factory(d_text=384)
    portable = RelationalTransformer(fp8_path, backend="torch", device="cpu")
    triton_model = RelationalTransformer(fp8_path, backend="triton")

    expected = portable.predict(context, activation="identity")
    actual = triton_model.predict(context, activation="identity")

    assert actual == pytest.approx(expected, abs=5e-2)


def test_triton_compiled_full_finetuning_updates_weights(
    tiny_checkpoint, customer_context_factory, tmp_path
):
    model = RelationalTransformer(tiny_checkpoint, device="cuda")
    before = model.model.dec_dict["number"].weight.detach().clone()
    examples = [
        RelationalExample(customer_context_factory(customer_id=501 + index), index % 2)
        for index in range(2)
    ]
    trainer = RelationalTrainer(
        model=model,
        args=RelationalTrainingArguments(
            output_dir=str(tmp_path / "triton-finetuned"),
            training_backend="triton",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            learning_rate=1e-3,
        ),
        train_dataset=examples,
    )

    result = trainer.train()

    assert result["steps"] == 1
    assert torch.isfinite(torch.tensor(result["train_loss"]))
    assert not torch.equal(before, model.model.dec_dict["number"].weight)
    assert (tmp_path / "triton-finetuned" / "model.safetensors").is_file()

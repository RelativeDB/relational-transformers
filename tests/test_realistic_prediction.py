from __future__ import annotations

import numpy as np
import pytest
import torch

from relational_transformers import RelationalTransformer


def test_typed_customer_context_preserves_schema_and_wide_ids(customer_context_factory):
    batch = customer_context_factory()

    assert batch.sequence_length == 13
    assert int(batch.node_idxs.max()) > 2**24
    assert len(torch.unique(batch.node_idxs)) == 4
    assert batch.number_values.shape == (1, 13, 1)
    assert batch.datetime_values.shape == (1, 13, 1)
    assert batch.boolean_values.shape == (1, 13, 1)
    assert batch.is_targets.sum().item() == 1
    assert batch.f2p_nbr_idxs[0, 5, 0].item() == 44_903_103


def test_batched_predictions_match_each_context_scored_alone(
    tiny_checkpoint, customer_context_factory
):
    model = RelationalTransformer(tiny_checkpoint, device="cpu")
    active = customer_context_factory()
    quiet = customer_context_factory(
        customer_id=44_903_207,
        plan="starter monthly",
        order_amounts=(18.0,),
        support_summary=None,
    )

    separate = np.asarray(
        [
            model.predict(active, activation="identity"),
            model.predict(quiet, activation="identity"),
        ]
    )
    batched = model.predict([active, quiet], activation="identity")

    np.testing.assert_allclose(batched, separate, rtol=1e-6, atol=1e-6)


def test_explicit_support_ablation_changes_prediction_without_mutating_input(
    tiny_checkpoint, customer_context_factory
):
    model = RelationalTransformer(tiny_checkpoint, device="cpu")
    context = customer_context_factory()
    original_padding = context.is_padding.clone()
    without_support_ticket = context.ablate([11, 12])

    full, ablated = model.predict(
        [context, without_support_ticket],
        activation="identity",
    )

    assert full != pytest.approx(ablated, abs=1e-7)
    assert torch.equal(context.is_padding, original_padding)
    assert without_support_ticket.is_padding[0, 11:].tolist() == [True, True]


def test_output_views_are_internally_consistent(tiny_checkpoint, customer_context_factory):
    model = RelationalTransformer(tiny_checkpoint, device="cpu")
    context = customer_context_factory()

    embeddings = model.forward(context, output="embeddings").embeddings
    features = model.forward(context, output="target_features").features
    tokens = model.forward(context, output="token_scores").token_scores
    score_and_text = model.forward(context, output="target_scores_and_text")
    target_mask = context.is_targets

    expected_features = (embeddings * target_mask.unsqueeze(-1)).sum(1)
    expected_scores = (tokens * target_mask).sum(1)
    torch.testing.assert_close(features, expected_features)
    torch.testing.assert_close(score_and_text.scores, expected_scores)
    assert score_and_text.target_text.shape == (1, context.d_text)


def test_relativedb_alias_mapping_matches_canonical_batch(
    tiny_checkpoint, customer_context_factory
):
    model = RelationalTransformer(tiny_checkpoint, device="cpu")
    context = customer_context_factory()
    canonical = context.numpy()
    aliases = {
        "node_idxs": canonical["node_idxs"],
        "f2p": canonical["f2p_nbr_idxs"],
        "col_idxs": canonical["col_name_idxs"],
        "table_idxs": canonical["table_name_idxs"],
        "is_padding": canonical["is_padding"],
        "sem_types": canonical["sem_types"],
        "is_target": canonical["is_targets"],
        "number_v": canonical["number_values"].squeeze(-1),
        "datetime_v": canonical["datetime_values"].squeeze(-1),
        "boolean_v": canonical["boolean_values"].squeeze(-1),
        "text_v": canonical["text_values"],
        "col_name_v": canonical["col_name_values"],
    }

    expected = model.predict(context, activation="identity")
    actual = model.predict(aliases, activation="identity")

    assert actual == pytest.approx(expected, abs=1e-7)


def test_classification_and_regression_apply_different_default_activations(
    tiny_checkpoint, customer_context_factory
):
    context = customer_context_factory()
    classifier = RelationalTransformer(tiny_checkpoint, task="classification", device="cpu")
    regressor = RelationalTransformer(tiny_checkpoint, task="regression", device="cpu")

    logit = classifier.predict(context, activation="identity")
    probability = classifier.predict(context)
    regression = regressor.predict(context)

    assert probability == pytest.approx(1.0 / (1.0 + np.exp(-logit)), rel=1e-6)
    assert 0.0 < probability < 1.0
    assert regression == pytest.approx(logit, abs=1e-7)


def test_saved_model_reloads_with_identical_prediction(
    tiny_checkpoint, customer_context_factory, tmp_path
):
    context = customer_context_factory()
    model = RelationalTransformer(tiny_checkpoint, device="cpu")
    expected = model.predict(context, activation="identity")

    destination = tmp_path / "customer-risk-model"
    model.save_pretrained(destination)
    restored = RelationalTransformer(destination, device="cpu")

    assert restored.predict(context, activation="identity") == pytest.approx(expected, abs=1e-7)


def test_meta_model_exposes_architecture_without_materializing_weights(tiny_checkpoint):
    model = RelationalTransformer(tiny_checkpoint, backend="meta")

    assert model.get_model_kwargs() == {
        "num_blocks": 2,
        "d_model": 12,
        "d_text": 4,
        "num_heads": 3,
        "d_ff": 24,
    }
    assert all(parameter.device.type == "meta" for parameter in model.model.parameters())
    with pytest.raises(RuntimeError, match="cannot run inference"):
        model.predict(np.ones((2, 8), np.float32), target=0)

from __future__ import annotations

import numpy as np
import pytest

from relational_transformers import RelationalBatch, RelationalTransformer


def test_missing_relational_field_names_the_problem(customer_context_factory):
    values = customer_context_factory().numpy()
    del values["f2p_nbr_idxs"]

    with pytest.raises(ValueError, match="missing fields: f2p_nbr_idxs"):
        RelationalBatch.from_mapping(values)


@pytest.mark.parametrize("channel", ["number_values", "datetime_values", "boolean_values"])
def test_nonfinite_scalar_channels_are_rejected(customer_context_factory, channel):
    values = customer_context_factory().numpy()
    values[channel][0, 1] = np.nan

    with pytest.raises(ValueError, match="scalar channels must be finite"):
        RelationalBatch.from_mapping(values)


@pytest.mark.parametrize("channel", ["text_values", "col_name_values"])
def test_nonfinite_embedding_channels_are_rejected(customer_context_factory, channel):
    values = customer_context_factory().numpy()
    values[channel][0, 3, 0] = np.inf

    with pytest.raises(ValueError, match="embedding channels must be finite"):
        RelationalBatch.from_mapping(values)


def test_invalid_semantic_type_is_rejected(customer_context_factory):
    values = customer_context_factory().numpy()
    values["sem_types"][0, 2] = 9

    with pytest.raises(ValueError, match="sem_types values must be in"):
        RelationalBatch.from_mapping(values)


def test_target_cannot_also_be_padding(customer_context_factory):
    values = customer_context_factory().numpy()
    values["is_padding"][0, 0] = True

    with pytest.raises(ValueError, match="padding cell cannot be a target"):
        RelationalBatch.from_mapping(values)


def test_text_cell_convenience_requires_even_embedding_width():
    with pytest.raises(ValueError, match=r"2\*d_text"):
        RelationalBatch.from_text_cells(np.ones((3, 7), np.float32), target=0)


def test_text_cell_convenience_rejects_more_than_five_parents():
    with pytest.raises(ValueError, match="more than 5 parents"):
        RelationalBatch.from_text_cells(
            np.ones((3, 8), np.float32),
            target=0,
            parents={1: [10, 11, 12, 13, 14, 15]},
        )


def test_ablation_rejects_target_cell(customer_context_factory):
    with pytest.raises(ValueError, match="target cells cannot be ablated"):
        customer_context_factory().ablate([0])


def test_vector_prediction_requires_a_target(tiny_checkpoint):
    model = RelationalTransformer(tiny_checkpoint, device="cpu")

    with pytest.raises(ValueError, match="target is required"):
        model.predict(np.ones((3, 8), np.float32))


def test_empty_batch_sequence_is_rejected(tiny_checkpoint):
    model = RelationalTransformer(tiny_checkpoint, device="cpu")

    with pytest.raises(TypeError, match="non-empty sequence"):
        model.predict([])


def test_collation_rejects_different_embedding_spaces(tiny_checkpoint):
    model = RelationalTransformer(tiny_checkpoint, device="cpu")
    four_wide = RelationalBatch.from_text_cells(np.ones((3, 8), np.float32), target=0)
    five_wide = RelationalBatch.from_text_cells(np.ones((3, 10), np.float32), target=0)

    with pytest.raises(ValueError, match="same d_text"):
        model.predict([four_wide, five_wide])

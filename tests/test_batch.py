import numpy as np
import pytest
import torch

from relational_transformers import RelationalBatch


def test_text_cell_convenience_builds_full_contract():
    cells = np.arange(4 * 8, dtype=np.float32).reshape(4, 8)
    batch = RelationalBatch.from_text_cells(cells, target=0, node_idxs=[0, 0, 1, 1])

    assert batch.node_idxs.shape == (1, 4)
    assert batch.text_values.shape == (1, 4, 4)
    assert batch.col_name_values.shape == (1, 4, 4)
    assert batch.is_targets.tolist() == [[True, False, False, False]]
    assert torch.equal(batch.col_name_values[0], torch.from_numpy(cells[:, :4]))


def test_ablation_pads_position_and_rejects_target():
    batch = RelationalBatch.from_text_cells(np.ones((3, 8), np.float32), target=0)
    ablated = batch.ablate([2])
    assert ablated.is_padding.tolist() == [[False, False, True]]
    assert not batch.is_padding.any()
    with pytest.raises(ValueError, match="target"):
        batch.ablate([0])


def test_mapping_accepts_flat_scalar_channels():
    batch = RelationalBatch.from_text_cells(np.ones((3, 8), np.float32), target=0)
    values = batch.numpy()
    values["number_values"] = values["number_values"].squeeze(-1)
    values["datetime_values"] = values["datetime_values"].squeeze(-1)
    values["boolean_values"] = values["boolean_values"].squeeze(-1)

    normalized = RelationalBatch.from_mapping(values)

    assert normalized.number_values.shape == (1, 3, 1)
    assert normalized.datetime_values.shape == (1, 3, 1)
    assert normalized.boolean_values.shape == (1, 3, 1)

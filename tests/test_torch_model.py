import numpy as np
import torch

from relational_transformers import RelationalBatch, RTJModel


def tiny_batch():
    rng = np.random.default_rng(4)
    return RelationalBatch.from_text_cells(
        rng.normal(size=(5, 8)).astype("float32"), target=0, node_idxs=range(5)
    )


def test_torch_outputs_cover_serving_contract():
    torch.manual_seed(2)
    model = RTJModel(num_blocks=2, d_model=8, d_text=4, num_heads=2, d_ff=16)
    batch = tiny_batch()

    assert model(batch).scores.shape == (1,)
    assert model(batch, "token_scores").token_scores.shape == (1, 5)
    assert model(batch, "target_features").features.shape == (1, 8)
    output = model(batch, "target_scores_and_text")
    assert output.scores.shape == (1,)
    assert output.target_text.shape == (1, 4)
    assert model(batch, "embeddings").embeddings.shape == (1, 5, 8)


def test_relations_change_prediction():
    torch.manual_seed(3)
    model = RTJModel(num_blocks=1, d_model=8, d_text=4, num_heads=2, d_ff=16)
    batch = tiny_batch()
    baseline_mask = model._masks(batch)["feat"].clone()
    batch.f2p_nbr_idxs[0, 0, 0] = 1
    changed_mask = model._masks(batch)["feat"]

    assert not torch.equal(baseline_mask, changed_mask)
    assert torch.isfinite(model(batch).scores).all()

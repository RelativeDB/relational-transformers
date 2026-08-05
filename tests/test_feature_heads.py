from __future__ import annotations

import numpy as np
import pytest
import torch

from relational_transformers import FineTunedHead, fit_feature_head

D_MODEL = 512


def test_multiclass_head_fits_saves_and_reloads(tmp_path):
    rng = np.random.default_rng(3)
    labels = np.asarray([0, 1, 2] * 15, np.float32)
    features = rng.normal(size=(45, D_MODEL)).astype(np.float32)
    for k in range(3):
        features[labels == k, k] += 1.5

    torch.manual_seed(5)
    decoder = torch.nn.Linear(D_MODEL, 384)
    class_embeddings = rng.normal(size=(3, 384)).astype(np.float32)
    class_embeddings /= np.linalg.norm(class_embeddings, axis=1, keepdims=True)

    head = fit_feature_head(features, labels, "multiclass",
                            classes=["yes", "no", "maybe"], epochs=120,
                            class_embeddings=class_embeddings,
                            text_decoder=decoder)
    assert head.n_outputs == 3
    assert head.initial_loss >= head.final_loss

    path = tmp_path / "head.safetensors"
    head.save(str(path))
    reloaded = FineTunedHead.load(str(path))
    assert reloaded.task == "multiclass"
    assert reloaded.classes == ("yes", "no", "maybe")
    logits = reloaded.predict(features)
    assert logits.shape == (45, 3)
    assert (logits.argmax(1) == labels).mean() > 0.8


def test_head_load_requires_sidecar(tmp_path):
    rng = np.random.default_rng(7)
    labels = np.asarray([0, 1] * 10, np.float32)
    features = rng.normal(size=(20, D_MODEL)).astype(np.float32)
    head = fit_feature_head(features, labels, "binary", epochs=5)
    path = tmp_path / "head.safetensors"
    head.save(str(path))
    (tmp_path / "head.safetensors.preproc.json").unlink()
    with pytest.raises(Exception, match="preproc"):
        FineTunedHead.load(str(path))


def test_ranking_head_learns_group_ordering():
    rng = np.random.default_rng(11)
    groups, per_group = 12, 4
    features = rng.normal(size=(groups * per_group, D_MODEL)).astype(np.float32)
    labels = np.zeros(groups * per_group, np.float32)
    for g in range(groups):
        labels[g * per_group] = 1.0
        features[g * per_group] += 0.6
    offsets = np.arange(0, groups * per_group + 1, per_group, dtype=np.int64)

    head = fit_feature_head(features, labels, "ranking",
                            group_offsets=offsets, n_groups=groups, epochs=80)
    logits = head.predict(features)[:, 0]
    winners = [int(np.argmax(logits[s:s + per_group])) == 0
               for s in range(0, groups * per_group, per_group)]
    assert sum(winners) >= groups - 2


def test_unknown_task_rejected():
    with pytest.raises(ValueError, match="task must be one of"):
        fit_feature_head(np.zeros((2, D_MODEL), np.float32),
                         np.zeros(2, np.float32), "survival")

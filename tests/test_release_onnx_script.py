from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _release_script():
    path = Path(__file__).parents[1] / "scripts" / "export_release_onnx.py"
    spec = importlib.util.spec_from_file_location("export_release_onnx", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_release_onnx_context_is_deterministic_and_masks_target_value():
    module = _release_script()
    first = module.example_context(cells=5, seed=13)
    second = module.example_context(cells=5, seed=13)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (5, 768)
    assert first.dtype == np.float32
    assert not np.all(first[0, :384] == 0)
    assert np.all(first[0, 384:] == 0)

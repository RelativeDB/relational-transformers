from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
EXAMPLES = ROOT / "examples"


class _Encoder:
    def encode(self, texts, *, convert_to_numpy=True):
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for text in texts:
            seed = sum(str(text).encode())
            rng = np.random.default_rng(seed)
            rows.append(rng.normal(size=384).astype(np.float32))
        return np.stack(rows)


def _common_module():
    spec = importlib.util.spec_from_file_location("example_common", EXAMPLES / "_common.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_every_example_is_valid_python_and_linked_from_example_readme():
    readme = (EXAMPLES / "README.md").read_text()
    scripts = sorted(path for path in EXAMPLES.glob("*.py") if not path.name.startswith("_"))

    assert len(scripts) >= 10
    for script in scripts:
        ast.parse(script.read_text(), filename=str(script))
        assert script.name in readme


def test_shared_builders_create_model_ready_vectors_and_relations():
    common = _common_module()
    encoder = _Encoder()
    cells = common.issue_cells(encoder, title="Timeout", body="Pool hangs")
    context = common.customer_context(
        encoder,
        customer_id=101,
        age=42,
        tenure_months=19,
        plan="business annual",
        order_amount=129,
        support_summary="Invoices fail after a card update.",
    )

    assert cells.shape == (3, 768)
    assert context.batch_size == 1
    assert context.sequence_length == 6
    assert context.is_targets.tolist() == [[True, False, False, False, False, False]]
    assert context.f2p_nbr_idxs[0, 4, 0].item() == 101
    assert context.f2p_nbr_idxs[0, 5, 0].item() == 101

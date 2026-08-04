from __future__ import annotations

import hashlib
from collections.abc import Callable

import numpy as np
import pytest
import torch

from relational_transformers import RelationalBatch, RTJModel
from relational_transformers.checkpoints import save_checkpoint
from relational_transformers.constants import SEM_BOOLEAN, SEM_DATETIME, SEM_NUMBER, SEM_TEXT


def _embedding(text: str, width: int) -> np.ndarray:
    """Stand-in for an application encoder with stable, nontrivial vectors."""

    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
    vector = np.random.default_rng(seed).normal(size=width).astype(np.float32)
    return vector / np.linalg.norm(vector)


def build_customer_context(
    *,
    customer_id: int = 44_903_103,
    age: float = 42.0,
    tenure_months: float = 19.0,
    plan: str = "business annual",
    marketing_opt_in: bool = True,
    order_amounts: tuple[float, ...] = (129.0, 58.5),
    support_summary: str | None = "Asked why invoices fail after card updates",
    d_text: int = 4,
) -> RelationalBatch:
    """Build one realistic customer context with orders and a support ticket."""

    cells: list[dict] = [
        {"node": customer_id, "table": 0, "column": 0, "semantic": SEM_BOOLEAN, "target": True},
        {"node": customer_id, "table": 0, "column": 1, "semantic": SEM_NUMBER, "number": age},
        {
            "node": customer_id,
            "table": 0,
            "column": 2,
            "semantic": SEM_NUMBER,
            "number": tenure_months,
        },
        {"node": customer_id, "table": 0, "column": 3, "semantic": SEM_TEXT, "text": plan},
        {
            "node": customer_id,
            "table": 0,
            "column": 4,
            "semantic": SEM_BOOLEAN,
            "boolean": float(marketing_opt_in),
        },
    ]
    for offset, amount in enumerate(order_amounts):
        order_id = customer_id + 100 + offset
        cells.extend(
            [
                {
                    "node": order_id,
                    "parents": [customer_id],
                    "table": 1,
                    "column": 5,
                    "semantic": SEM_NUMBER,
                    "number": amount,
                },
                {
                    "node": order_id,
                    "parents": [customer_id],
                    "table": 1,
                    "column": 6,
                    "semantic": SEM_DATETIME,
                    "datetime": 0.82 - offset * 0.07,
                },
                {
                    "node": order_id,
                    "parents": [customer_id],
                    "table": 1,
                    "column": 7,
                    "semantic": SEM_TEXT,
                    "text": "cloud storage" if offset == 0 else "team seats",
                },
            ]
        )
    if support_summary is not None:
        ticket_id = customer_id + 1_000
        cells.extend(
            [
                {
                    "node": ticket_id,
                    "parents": [customer_id],
                    "table": 2,
                    "column": 8,
                    "semantic": SEM_TEXT,
                    "text": support_summary,
                },
                {
                    "node": ticket_id,
                    "parents": [customer_id],
                    "table": 2,
                    "column": 9,
                    "semantic": SEM_DATETIME,
                    "datetime": 0.96,
                },
            ]
        )

    size = len(cells)
    parents = np.full((1, size, 5), -1, dtype=np.int64)
    number = np.zeros((1, size), dtype=np.float32)
    datetime = np.zeros((1, size), dtype=np.float32)
    boolean = np.zeros((1, size), dtype=np.float32)
    text = np.zeros((1, size, d_text), dtype=np.float32)
    columns = np.zeros((1, size, d_text), dtype=np.float32)
    for position, cell in enumerate(cells):
        cell_parents = cell.get("parents", [])
        parents[0, position, : len(cell_parents)] = cell_parents
        number[0, position] = cell.get("number", 0.0)
        datetime[0, position] = cell.get("datetime", 0.0)
        boolean[0, position] = cell.get("boolean", 0.0)
        if "text" in cell:
            text[0, position] = _embedding(cell["text"], d_text)
        columns[0, position] = _embedding(f"table {cell['table']} column {cell['column']}", d_text)

    return RelationalBatch(
        node_idxs=np.asarray([[cell["node"] for cell in cells]], dtype=np.int64),
        f2p_nbr_idxs=parents,
        col_name_idxs=np.asarray([[cell["column"] for cell in cells]], dtype=np.int64),
        table_name_idxs=np.asarray([[cell["table"] for cell in cells]], dtype=np.int64),
        is_padding=np.zeros((1, size), dtype=bool),
        sem_types=np.asarray([[cell["semantic"] for cell in cells]], dtype=np.int64),
        is_targets=np.asarray([[cell.get("target", False) for cell in cells]], dtype=bool),
        number_values=number,
        datetime_values=datetime,
        boolean_values=boolean,
        text_values=text,
        col_name_values=columns,
    )


@pytest.fixture
def customer_context_factory() -> Callable[..., RelationalBatch]:
    return build_customer_context


@pytest.fixture
def tiny_checkpoint(tmp_path):
    torch.manual_seed(17)
    model = RTJModel(num_blocks=2, d_model=12, d_text=4, num_heads=3, d_ff=24)
    config = {
        "task_type": "clf",
        "model": {"num_blocks": 2, "d_model": 12, "d_text": 4, "num_heads": 3, "d_ff": 24},
    }
    save_checkpoint(model, tmp_path, config)
    return tmp_path

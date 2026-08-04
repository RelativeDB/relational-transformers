"""Shared data builders for the runnable examples."""

from __future__ import annotations

import numpy as np

from relational_transformers import RelationalBatch


def issue_cells(encoder, *, title: str, body: str, comment: str | None = None) -> np.ndarray:
    """Encode an issue as ``[column embedding, value embedding]`` vectors."""

    columns = ["is bug", "title", "body"]
    values = [None, title, body]
    if comment is not None:
        columns.append("latest comment")
        values.append(comment)
    column_vectors = encoder.encode(columns, convert_to_numpy=True).astype(np.float32)
    value_vectors = np.zeros_like(column_vectors)
    present = [index for index, value in enumerate(values) if value is not None]
    value_vectors[present] = encoder.encode(
        [values[index] for index in present], convert_to_numpy=True
    ).astype(np.float32)
    return np.concatenate([column_vectors, value_vectors], axis=1)


def customer_context(
    encoder,
    *,
    customer_id: int,
    age: float,
    tenure_months: float,
    plan: str,
    order_amount: float,
    support_summary: str | None,
) -> RelationalBatch:
    """Build a typed customer/order/support context with real relations."""

    cells = [
        (customer_id, -1, 0, 0, 3, True, None),
        (customer_id, -1, 0, 1, 0, False, age / 100.0),
        (customer_id, -1, 0, 2, 0, False, tenure_months / 120.0),
        (customer_id, -1, 0, 3, 1, False, plan),
        (customer_id + 100, customer_id, 1, 4, 0, False, order_amount / 500.0),
    ]
    if support_summary is not None:
        cells.append((customer_id + 200, customer_id, 2, 5, 1, False, support_summary))

    size = len(cells)
    parents = np.full((1, size, 5), -1, dtype=np.int64)
    number = np.zeros((1, size, 1), dtype=np.float32)
    text = np.zeros((1, size, 384), dtype=np.float32)
    column_names = ["will churn", "age", "tenure months", "plan", "order amount"]
    if support_summary is not None:
        column_names.append("support summary")
    columns = encoder.encode(column_names, convert_to_numpy=True).astype(np.float32)[None, :, :]

    text_positions = []
    text_values = []
    for position, (_, parent, _, _, semantic, _, value) in enumerate(cells):
        if parent >= 0:
            parents[0, position, 0] = parent
        if semantic == 0 and value is not None:
            number[0, position, 0] = value
        elif semantic == 1 and value is not None:
            text_positions.append(position)
            text_values.append(value)
    if text_values:
        text[0, text_positions] = encoder.encode(text_values, convert_to_numpy=True).astype(
            np.float32
        )

    return RelationalBatch(
        node_idxs=np.asarray([[cell[0] for cell in cells]], dtype=np.int64),
        f2p_nbr_idxs=parents,
        col_name_idxs=np.asarray([[cell[3] for cell in cells]], dtype=np.int64),
        table_name_idxs=np.asarray([[cell[2] for cell in cells]], dtype=np.int64),
        is_padding=np.zeros((1, size), dtype=bool),
        sem_types=np.asarray([[cell[4] for cell in cells]], dtype=np.int64),
        is_targets=np.asarray([[cell[5] for cell in cells]], dtype=bool),
        number_values=number,
        datetime_values=np.zeros((1, size, 1), dtype=np.float32),
        boolean_values=np.zeros((1, size, 1), dtype=np.float32),
        text_values=text,
        col_name_values=columns,
    )

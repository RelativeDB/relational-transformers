from __future__ import annotations

import numpy as np

from relational_transformers.backends.triton_data import build_worklists, sort_batch


def test_sort_batch_orders_schema_columns_and_leaves_padding_last(customer_context_factory):
    first = customer_context_factory(order_amounts=(19.0,), support_summary=None)
    second = customer_context_factory(customer_id=44_903_501)
    from relational_transformers.model import _collate_batches

    raw = _collate_batches([first, second]).numpy()
    sorted_batch, orders = sort_batch(raw)

    for row in range(2):
        valid = ~sorted_batch["is_padding"][row].astype(bool)
        columns = sorted_batch["col_name_idxs"][row, valid]
        assert np.all(columns[:-1] <= columns[1:])
        assert sorted_batch["is_padding"][row, valid.sum() :].all()
        assert sorted(orders[row].tolist()) == list(range(orders.shape[1]))


def test_worklists_include_column_feature_and_neighbor_attention(customer_context_factory):
    context = customer_context_factory().numpy()
    sorted_batch, _ = sort_batch(context)

    column, feature, neighbor = build_worklists(sorted_batch)

    assert column.qidx.size == 13
    assert feature.qidx.size == 13
    assert neighbor.qidx.size > 0
    assert neighbor.kidx.size > 0
    assert np.all(column.nk > 0)
    assert np.all(feature.nk > 0)
    assert np.all(neighbor.nk > 0)

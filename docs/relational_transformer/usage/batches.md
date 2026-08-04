# Relational batches

`RelationalBatch` is the stable contract shared by every backend and by RelativeDB's
native engine. Integer tensors describe topology, value tensors carry already encoded
model inputs, and two boolean masks mark padding and targets.

## Fields

| Field | Shape | Purpose |
|---|---|---|
| `node_idxs` | `[B, S]` | Row/node identity for each cell |
| `f2p_nbr_idxs` | `[B, S, 5]` | Foreign-key parent node identities, `-1` padded |
| `col_name_idxs` | `[B, S]` | Column vocabulary identity |
| `table_name_idxs` | `[B, S]` | Table vocabulary identity |
| `is_padding` | `[B, S]` | `True` marks a padded position |
| `sem_types` | `[B, S]` | `0` number, `1` text, `2` datetime, `3` boolean |
| `is_targets` | `[B, S]` | `True` marks a masked prediction cell |
| `number_values` | `[B, S, 1]` | Normalized numeric channel |
| `datetime_values` | `[B, S, 1]` | Normalized timestamp channel |
| `boolean_values` | `[B, S, 1]` | Boolean channel as floats |
| `text_values` | `[B, S, d_text]` | User-encoded text values |
| `col_name_values` | `[B, S, d_text]` | User-encoded column descriptions |

Node identities can be production-scale integers such as primary keys; the model compares
them for equality and never uses them as array indices. Column and table identities are
per-batch vocabulary IDs, so first-seen numbering within each batch is fine. A cell with
up to five foreign-key parents lists their node identities in `f2p_nbr_idxs` and pads the
rest with `-1`.

Cells stay in the order you provide them. Backends may reorder internally for speed, but
`token_scores` and `embeddings` always come back in caller order.

## How the Model Reads a Batch

Each cell's input state starts from its encoded column name. The value channel selected
by `sem_types` is added for regular cells. Target cells receive a learned mask embedding
for their semantic type, and their value channel is ignored. Attention then flows along
three relations derived from the index fields: cells of the same node or its parents,
cells whose key is a foreign-key parent of the query's node, and cells sharing a column
within a table. Padding is excluded from all three.

## Constructing a Batch

The constructor accepts numpy arrays, torch tensors, or nested lists and canonicalizes
them: index fields become `int64`, masks become `bool` (so `uint8` inputs work), value
channels become floating point, and `[B, S]` scalar channels gain the trailing axis.
Validation runs immediately and raises `ValueError` naming the first offending field.

`from_mapping` builds a batch from a dictionary and accepts RelativeDB's short aliases:

```python
from relational_transformers import RelationalBatch

batch = RelationalBatch.from_mapping({
    "node_idxs": node_idxs,
    "f2p": f2p,                      # alias of f2p_nbr_idxs
    "col_idxs": col_idxs,            # alias of col_name_idxs
    "table_idxs": table_idxs,        # alias of table_name_idxs
    "is_padding": is_padding,
    "sem_types": sem_types,
    "is_target": is_target,          # alias of is_targets
    "number_v": number_v,            # alias of number_values
    "datetime_v": datetime_v,
    "boolean_v": boolean_v,
    "text_v": text_v,
    "col_name_v": col_name_v,
})
```

`as_dict()` returns the canonical fields under their full names, so a batch round-trips
through `from_mapping(batch.as_dict())`. `numpy()` produces the same dictionary as CPU
numpy arrays, which is what the Triton backend consumes.

```{eval-rst}
.. collapse:: Validation rules

   Construction rejects a batch when:

   - ``node_idxs`` is not two-dimensional, or any per-cell field disagrees with its ``[B, S]`` shape;
   - ``f2p_nbr_idxs`` is not ``[B, S, 5]``;
   - a scalar channel is not ``[B, S, 1]`` after canonicalization;
   - ``col_name_values`` and ``text_values`` differ in shape;
   - any ``sem_types`` value falls outside ``[0, 3]``;
   - a cell is both padding and a target;
   - any value channel contains NaN or infinity.

   Encode a missing value as ``0.0`` with the correct semantic type; NaN fails validation.
```

## All-text Convenience Batches

`from_text_cells` builds a single-row batch from `[cells, 2*d_text]` arrays where each
row is a column embedding followed by a value embedding. Optional keywords supply node
identities, foreign-key parents, and table IDs. Everything is typed as text, so use it
for prototypes and text-only contexts, and build the full batch when scalar channels
matter.

```python
batch = RelationalBatch.from_text_cells(
    cells,
    target=0,
    node_idxs=[7, 7, 8],
    parents={2: [7]},
)
```

## Device Movement and Ablation

`batch.to(device)` returns a copy on the target device, with an optional dtype for the
floating-point channels. `batch.ablate(positions)` pads out the selected cells while
positions and node identities stay stable; the [Ablation](ablation.md) page builds on it.

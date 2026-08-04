# Relational batches

`RelationalBatch` is the stable contract shared by every backend.

| Field | Shape | Purpose |
|---|---|---|
| `node_idxs` | `[B, S]` | Row/node identity for each cell |
| `f2p_nbr_idxs` | `[B, S, 5]` | Foreign-key parent node identities |
| `col_name_idxs` | `[B, S]` | Column vocabulary identity |
| `table_name_idxs` | `[B, S]` | Table vocabulary identity |
| `is_padding` | `[B, S]` | Padded positions |
| `sem_types` | `[B, S]` | Number, text, datetime, or boolean |
| `is_targets` | `[B, S]` | Masked prediction cells |
| scalar values | `[B, S, 1]` | Number, datetime, and boolean channels |
| `text_values` | `[B, S, 384]` | User-encoded text values |
| `col_name_values` | `[B, S, 384]` | User-encoded schema descriptions |

All arrays are raw pre-sort order. Backends may reorder cells internally but
public token outputs preserve caller order.

`RelationalBatch.from_mapping` accepts the existing RelativeDB aliases such as
`f2p`, `col_idxs`, `number_v`, and `text_v`. This keeps integrations direct and
avoids copying or translating through Python cell objects.

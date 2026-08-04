# Dataset Overview

Relational Transformers trains on `RelationalExample` objects. Each example
contains one model-ready relational context and its label.

```python
from relational_transformers import RelationalDataset, RelationalExample

dataset = RelationalDataset([
    RelationalExample(input=customer_a_batch, label=1),
    RelationalExample(input=customer_b_batch, label=0),
])
```

## Accepted Input Types

An example input can be a `RelationalBatch`, a mapping containing the canonical
tensor fields, or a simple all-text cell-vector array. Production training
normally uses one unpadded `RelationalBatch` per example; the trainer pads and
collates variable context lengths.

## Dataset Construction

Use `RelationalDataset.from_inputs(inputs, labels)` when inputs and labels are
already stored separately. It validates aligned lengths and retains inputs
without serializing or re-encoding them.

## Splitting Relational Data

Split by time and entity before retrieving context. A random row split can put
future events, related copies of the same entity, or normalization statistics
from the evaluation period into training. Construct each context only from
information available at its anchor time.

## Pre-existing Datasets

There is no universal raw-table dataset format in this package because the
application owns retrieval and cell encoding. Persist canonical batch tensors
when reproducibility matters, and record the encoder identity and normalization
statistics alongside them.

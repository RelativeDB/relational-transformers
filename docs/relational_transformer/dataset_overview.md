# Dataset Overview

Training and evaluation consume `RelationalExample` objects. Each example pairs one
model-ready relational context with its label.

```python
from relational_transformers import RelationalDataset, RelationalExample

dataset = RelationalDataset([
    RelationalExample(input=customer_a_batch, label=1),
    RelationalExample(input=customer_b_batch, label=0),
])
```

`RelationalDataset` is a standard `torch.utils.data.Dataset`. It validates that every
item is a `RelationalExample`, rejects empty collections, and supports indexing,
iteration, and `len()`.

## Accepted Input Types

An example input can be anything `predict` accepts for a single context:

- a `RelationalBatch` with batch size 1, the normal production case;
- a mapping of canonical tensor fields, converted through `RelationalBatch.from_mapping`;
- an all-text `[cells, 2*d_text]` cell array for prototypes.

Contexts may differ in length. The trainer pads and collates them per mini-batch, so
store each example unpadded. Every context in a dataset must share the same `d_text` and
checkpoint contract.

Labels depend on the problem type: floats or 0/1 for `binary`, integer class IDs for
`multiclass`, multi-hot vectors for `multilabel`, and floats for `regression` and
`forecasting`. The [Loss Overview](loss_overview.md) lists the exact shapes.

## Dataset Construction

Use `RelationalDataset.from_inputs(inputs, labels)` when inputs and labels are already
stored separately. It validates aligned lengths and keeps inputs as-is without
serializing or re-encoding them.

```python
dataset = RelationalDataset.from_inputs(contexts, churn_labels)
```

A plain list of `RelationalExample` objects also works everywhere a dataset is accepted;
the trainer and evaluators only need iteration and indexing.

## Splitting Relational Data

Split by time and entity before retrieving context. A random row split can put future
events, related copies of the same entity, or normalization statistics from the
evaluation period into training, and the resulting metrics look better than the deployed
model will. Construct each context only from information available at its anchor time.

```{eval-rst}
.. collapse:: Leakage checklist

   - **Future events**: does any cell in a training context postdate its label's anchor time?
   - **Entity overlap**: does the same customer, order, or ticket appear in both splits through different rows?
   - **Normalization statistics**: were the means and standard deviations used for scalar channels computed on training rows only?
   - **Target echoes**: does a non-target cell encode the label, such as a status column written after the outcome?
```

## Pre-existing Datasets

There is no universal raw-table dataset format in this package because the application
owns retrieval and cell encoding. Persist canonical batch tensors when reproducibility
matters, and record the encoder identity and normalization statistics alongside them. A
saved dataset that has drifted from its encoder produces silently wrong features, so
treat the encoder version as part of the dataset.

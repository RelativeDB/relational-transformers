# Quickstart

Relational Transformers makes predictions over cells that your application has already
encoded into vectors. This page loads a pretrained model, builds a small context by hand,
and reads a prediction back out. Every later page assumes these steps.

## Load RT-J

```python
from relational_transformers import RelationalTransformer

model = RelationalTransformer()
```

The constructor downloads the checkpoint from the Hugging Face Hub on first use, picks
CUDA, MPS, or CPU automatically, and loads the classification checkpoint by default. Each
published repository also carries a regression checkpoint:

```python
regressor = RelationalTransformer("RelativeDB/rt-j-fp16", task="regression")
```

Pass `backend="triton"` or `backend="onnx"` to serve through the deployment backends and
`device="cpu"` or similar to override device selection. The [Backends](relational_transformer/usage/backends.md)
page covers all four backends.

## Encoding Cells

RT-J never sees strings. Each text cell is a vector with two channels: an embedding of the
column name followed by an embedding of the value. Your application chooses the encoder
and runs it. The released checkpoints were trained against
`sentence-transformers/all-MiniLM-L12-v2`, which produces 384-wide vectors, so a
model-ready text cell is 768 floats.

```{eval-rst}
.. sidebar:: Documentation

   #. :class:`~relational_transformers.RelationalTransformer`
   #. :meth:`RelationalTransformer.predict() <relational_transformers.RelationalTransformer.predict>`

   **Related links:**

   - `Prediction <relational_transformer/usage/prediction.html>`_
   - `Pretrained Models <relational_transformer/pretrained_models.html>`_
```

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from relational_transformers import RelationalTransformer

# 1. Load the model and the encoder your application uses for text cells
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
model = RelationalTransformer("RelativeDB/rt-j-fp16")

issue = {
    "title": "Database connections time out after 30 seconds",
    "body": "The pool stops returning connections after the service has been idle.",
    "latest_comment": "Restarting the process temporarily fixes it.",
}

def text_cell(column, value):
    return np.concatenate([encoder.encode(column), encoder.encode(value)])

def text_target(column):
    # The masked target keeps its column embedding and zeroes the value channel
    return np.concatenate([encoder.encode(column), np.zeros(384, np.float32)])

# 2. Stack the cells; row 0 is the masked prediction target
cells = np.stack([
    text_target("is bug"),
    text_cell("title", issue["title"]),
    text_cell("body", issue["body"]),
    text_cell("latest comment", issue["latest_comment"]),
])

# 3. Predict; target=0 marks which row the model fills in
probability = model.predict(cells, target=0)
print(f"P(issue is a bug) = {probability:.1%}")
```

`predict` builds a single-row batch from the array, marks row 0 as the target, runs the
model, and applies a sigmoid because the loaded checkpoint is a classifier. The return
value for one context is a plain Python float.

This convenience path treats every cell as text and gives them all the same node. Real
contexts carry numbers, datetimes, booleans, and relations between rows, which is what
`RelationalBatch` is for.

## Typed Cells

Do not stringify numbers, booleans, or datetimes. Each cell declares a semantic type, and
scalar values travel through their own channel after normalization in your feature
pipeline. The value of a target cell is ignored wherever `is_targets` is true.

```{eval-rst}
.. collapse:: A complete typed RelationalBatch

   The full tensor contract has twelve fields. This example builds one context with a
   customer row, one related order row, and a boolean churn target::

      import numpy as np
      from relational_transformers import RelationalBatch

      # cells:      0 target    1 age      2 plan     3 amount   4 placed_at
      # node:       customer    customer   customer   order      order
      # sem_types:  3 boolean   0 number   1 text     0 number   2 datetime
      batch = RelationalBatch(
          node_idxs=[[7, 7, 7, 8, 8]],
          f2p_nbr_idxs=[[[-1] * 5, [-1] * 5, [-1] * 5, [7, -1, -1, -1, -1], [7, -1, -1, -1, -1]]],
          col_name_idxs=[[0, 1, 2, 3, 4]],
          table_name_idxs=[[0, 0, 0, 1, 1]],
          is_padding=[[False] * 5],
          sem_types=[[3, 0, 1, 0, 2]],
          is_targets=[[True, False, False, False, False]],
          number_values=[[0.0, 0.31, 0.0, -0.52, 0.0]],       # normalized scalars
          datetime_values=[[0.0, 0.0, 0.0, 0.0, 0.82]],       # normalized timestamps
          boolean_values=[[0.0, 0.0, 0.0, 0.0, 0.0]],
          text_values=text_embeddings,                        # [1, 5, 384], zeros off text cells
          col_name_values=column_embeddings,                  # [1, 5, 384], one per cell
      )
      probability = model.predict(batch)

   The order cells list node ``7`` as their foreign-key parent, so the model attends from
   the order back to the customer. See `Relational Batches
   <relational_transformer/usage/batches.html>`_ for every field, shape, and convention.
```

An application integration retrieves rows, computes normalization statistics, embeds
text and column names, constructs the node and foreign-key arrays, and passes the
finished `RelationalBatch` into this library.

## Batch Prediction

Pass a list of cell arrays for simple all-text contexts, or one padded `RelationalBatch`
for typed contexts. Contexts in a list may have different lengths; the model pads and
collates them for you.

```python
probabilities = model.predict([customer_a, customer_b, customer_c], target=0)
probabilities = model.predict(relational_batch)
```

## Next Steps

- [Prediction](relational_transformer/usage/prediction.md) describes activations, logits, and every model output.
- [Relational Batches](relational_transformer/usage/batches.md) documents the full tensor contract.
- [Ablation](relational_transformer/usage/ablation.md) measures how much a group of cells affects a prediction.
- [Training Overview](relational_transformer/training_overview.md) covers head tuning and full fine-tuning.
- [Backends](relational_transformer/usage/backends.md) compares PyTorch, Triton, ONNX, and meta loading.

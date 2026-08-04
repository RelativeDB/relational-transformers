# Quickstart

## Load RT-J

```python
from relational_transformers import RelationalTransformer

model = RelationalTransformer("RelativeDB/rt-j-fp16")
```

Classification is the default checkpoint. Use `task="regression"` for the
regression/forecasting checkpoint.

## Encoding cells

RT-J text cells preserve two channels: a column-name embedding and a value
embedding. Your application chooses and runs the encoder. The released model
was trained against `all-MiniLM-L12-v2`.

```python
import numpy as np
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

issue = {
    "title": "Database connections time out after 30 seconds",
    "body": "The pool stops returning connections after the service has been idle.",
    "latest_comment": "Restarting the process temporarily fixes it.",
}

def text_cell(column, value):
    return np.concatenate([encoder.encode(column), encoder.encode(value)])

def text_target(column):
    return np.concatenate([encoder.encode(column), np.zeros(384, np.float32)])

cells = np.stack([
    text_target("is bug"),
    text_cell("title", issue["title"]),
    text_cell("body", issue["body"]),
    text_cell("latest comment", issue["latest_comment"]),
])
probability = model.predict(cells, target=0)
```

The standalone convenience path assumes one all-text row. Production contexts
normally use `RelationalBatch`, which retains typed scalar channels and graph
structure.

## Typed cells

Do not stringify numbers, booleans, or datetimes. Normalize them in the feature
pipeline and populate their scalar channel. Set `sem_types` to number, text,
datetime, or boolean. The target value is ignored wherever `is_targets=True`.

RelativeDB is the reference integration: it retrieves rows, computes
normalization statistics, embeds text and column names, constructs node and
foreign-key arrays, and passes the resulting `RelationalBatch` into this
library.

## Batch prediction

Pass a list of vector arrays for simple all-text contexts, or one padded
`RelationalBatch` for typed contexts:

```python
probabilities = model.predict([customer_a, customer_b, customer_c], target=0)
probabilities = model.predict(relational_batch)
```

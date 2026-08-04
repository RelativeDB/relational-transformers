"""Build text-cell vectors and make one RT-J classification prediction."""

import numpy as np
from sentence_transformers import SentenceTransformer

from relational_transformers import RelationalTransformer

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
model = RelationalTransformer("RelativeDB/rt-j-fp16")


def text_cell(column, value):
    return np.concatenate([encoder.encode(column), encoder.encode(value)])


cells = np.stack([
    np.concatenate([encoder.encode("is bug"), np.zeros(384, np.float32)]),
    text_cell("title", "Database connections time out after 30 seconds"),
    text_cell("body", "The pool stops returning connections after idle time"),
])
print(model.predict(cells, target=0))

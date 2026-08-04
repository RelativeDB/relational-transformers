"""Run batched native-FP8 inference through the Triton CUDA backend."""

from _common import issue_cells
from sentence_transformers import SentenceTransformer

from relational_transformers import RelationalTransformer

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
model = RelationalTransformer("RelativeDB/rt-j-fp8", backend="triton", device="cuda")
batch = [
    issue_cells(
        encoder,
        title="Connection timeout",
        body="The pool hangs after the service is idle.",
    ),
    issue_cells(
        encoder,
        title="Document retries",
        body="Explain retry configuration and defaults.",
    ),
]
print(model.predict(batch, target=0))

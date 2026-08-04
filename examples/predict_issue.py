"""Build text-cell vectors and make one RT-J classification prediction."""

from _common import issue_cells
from sentence_transformers import SentenceTransformer

from relational_transformers import RelationalTransformer

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
model = RelationalTransformer("RelativeDB/rt-j-fp16")

cells = issue_cells(
    encoder,
    title="Database connections time out after 30 seconds",
    body="The pool stops returning connections after idle time",
    comment="Restarting the process temporarily fixes it.",
)
probability = model.predict(cells, target=0)
assert 0.0 <= probability <= 1.0
print(f"P(issue is a bug) = {probability:.1%}")

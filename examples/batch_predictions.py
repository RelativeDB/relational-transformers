"""Score variable-length issue contexts together in one model call."""

from _common import issue_cells
from sentence_transformers import SentenceTransformer

from relational_transformers import RelationalTransformer

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
model = RelationalTransformer("RelativeDB/rt-j-fp16")

issues = [
    issue_cells(
        encoder,
        title="Database connections time out after 30 seconds",
        body="The pool stops returning connections after idle time.",
        comment="Restarting the process temporarily fixes it.",
    ),
    issue_cells(
        encoder,
        title="Document the new retry configuration",
        body="The deployment guide should include the available retry modes.",
    ),
    issue_cells(
        encoder,
        title="Incorrect total after applying two coupons",
        body="Checkout displays a negative subtotal when discounts are stacked.",
        comment="Reproduced on the current release with a new account.",
    ),
]

for index, probability in enumerate(model.predict(issues, target=0), start=1):
    print(f"issue {index}: P(bug)={probability:.1%}")

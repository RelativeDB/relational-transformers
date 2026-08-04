"""Fit a lightweight multiclass head over frozen relational features."""

import os

import numpy as np
from _common import issue_cells
from sentence_transformers import SentenceTransformer

from relational_transformers import RelationalExample, RelationalTransformer

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
model = RelationalTransformer("RelativeDB/rt-j-fp16")
training_rows = [
    ("Connection timeout", "Pool hangs after idle time", 0),
    ("Negative checkout total", "Two coupons produce an invalid subtotal", 0),
    ("Add SAML login", "Enterprise customers need SSO", 1),
    ("Support CSV exports", "Allow reports to download as CSV", 1),
    ("Document retries", "Explain retry configuration and defaults", 2),
    ("Update migration guide", "Show the v2 schema migration", 2),
]
examples = [
    RelationalExample(issue_cells(encoder, title=title, body=body), label, target=0)
    for title, body, label in training_rows
]
head = model.fit_head(
    examples,
    task="issue_type",
    num_labels=3,
    problem_type="multiclass",
    epochs=int(os.environ.get("RT_EXAMPLE_EPOCHS", "100")),
    learning_rate=1e-3,
)
head.save_pretrained("models/issue-type-head")

test_issue = issue_cells(
    encoder,
    title="OAuth callback intermittently fails",
    body="Login returns a state mismatch after the session sits idle.",
)
probabilities = model.predict(test_issue, target=0, task_head="issue_type")
assert probabilities.shape == (1, 3)
assert np.isfinite(probabilities).all()
np.testing.assert_allclose(probabilities.sum(axis=1), [1.0], rtol=1e-5, atol=1e-5)
print(dict(zip(("bug", "feature", "documentation"), probabilities[0], strict=True)))

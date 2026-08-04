"""Evaluate classification quality and named context ablations."""

import numpy as np
from _common import customer_context
from sentence_transformers import SentenceTransformer

from relational_transformers import (
    BinaryClassificationEvaluator,
    RelationalExample,
    RelationalTransformer,
)

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
model = RelationalTransformer("RelativeDB/rt-j-fp16")
rows = [
    (101, 13, "starter monthly", 18, "Unable to export invoices.", 1),
    (102, 72, "business annual", 310, None, 0),
    (103, 4, "starter monthly", 9, "Asked to cancel after an outage.", 1),
    (104, 48, "business annual", 240, None, 0),
]
examples = [
    RelationalExample(
        input=customer_context(
            encoder,
            customer_id=customer_id,
            age=40,
            tenure_months=tenure,
            plan=plan,
            order_amount=amount,
            support_summary=support or "No recent support request.",
        ),
        label=label,
    )
    for customer_id, tenure, plan, amount, support, label in rows
]

metrics = BinaryClassificationEvaluator(examples)(model)
assert 0.0 <= metrics["accuracy"] <= 1.0
assert 0.0 <= metrics["precision"] <= 1.0
assert 0.0 <= metrics["recall"] <= 1.0
assert 0.0 <= metrics["f1"] <= 1.0

# Named ablations: pad out cell groups and compare identity-activation scores.
# relational-transformers-utils packages this workflow as AblationEvaluator.
inputs = [example.input for example in examples]
baseline = np.asarray(model.predict(inputs, activation="identity")).reshape(-1)
for name, positions in {"support": [5], "latest_order": [4]}.items():
    ablated = [batch.ablate(positions) for batch in inputs]
    changed = np.asarray(model.predict(ablated, activation="identity")).reshape(-1)
    difference = changed - baseline
    metrics[f"{name}_mean_delta"] = float(difference.mean())
    metrics[f"{name}_mean_absolute_delta"] = float(np.abs(difference).mean())

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

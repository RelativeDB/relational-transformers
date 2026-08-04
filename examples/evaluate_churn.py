"""Evaluate classification quality and named context ablations."""

from _common import customer_context
from sentence_transformers import SentenceTransformer

from relational_transformers import (
    AblationEvaluator,
    BinaryClassificationEvaluator,
    RelationalExample,
    RelationalTransformer,
    SequentialEvaluator,
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
evaluator = SequentialEvaluator(
    [
        BinaryClassificationEvaluator(examples),
        AblationEvaluator(examples, {"support": [5], "latest_order": [4]}),
    ]
)
metrics = evaluator(model)
assert 0.0 <= metrics["accuracy"] <= 1.0
assert 0.0 <= metrics["precision"] <= 1.0
assert 0.0 <= metrics["recall"] <= 1.0
assert 0.0 <= metrics["f1"] <= 1.0
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

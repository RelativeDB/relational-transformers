"""Measure how removing support history changes a churn prediction."""

from _common import customer_context
from sentence_transformers import SentenceTransformer

from relational_transformers import RelationalTransformer

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
model = RelationalTransformer("RelativeDB/rt-j-fp16")
context = customer_context(
    encoder,
    customer_id=44_903_103,
    age=42,
    tenure_months=19,
    plan="business annual",
    order_amount=129,
    support_summary="Invoices fail after the saved card is updated.",
)

# Cell 5 is the support summary. The caller chooses the ablation explicitly.
full, without_support = model.predict([context, context.ablate([5])])
assert 0.0 <= full <= 1.0
assert 0.0 <= without_support <= 1.0
print(f"full context:    {full:.1%}")
print(f"without support: {without_support:.1%}")
print(f"change:          {without_support - full:+.1%}")

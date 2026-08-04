"""Predict churn from typed customer, order, and support cells."""

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

print(f"P(customer churns) = {model.predict(context):.1%}")
print("cells:", context.sequence_length, "batch size:", context.batch_size)

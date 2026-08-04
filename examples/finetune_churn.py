"""Fine-tune the complete transformer on a small churn dataset."""

from _common import customer_context
from sentence_transformers import SentenceTransformer

from relational_transformers import (
    RelationalExample,
    RelationalTrainer,
    RelationalTrainingArguments,
    RelationalTransformer,
)

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
model = RelationalTransformer("RelativeDB/rt-j-fp16")
rows = [
    (201, 3, "starter monthly", 12, "Asked to cancel after an outage.", 1.0),
    (202, 61, "business annual", 340, None, 0.0),
    (203, 8, "starter monthly", 22, "Card has failed three times.", 1.0),
    (204, 44, "business annual", 180, None, 0.0),
]
dataset = [
    RelationalExample(
        customer_context(
            encoder,
            customer_id=customer_id,
            age=40,
            tenure_months=tenure,
            plan=plan,
            order_amount=amount,
            support_summary=support,
        ),
        label,
    )
    for customer_id, tenure, plan, amount, support, label in rows
]
trainer = RelationalTrainer(
    model=model,
    args=RelationalTrainingArguments(
        output_dir="models/churn",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        learning_rate=2e-5,
    ),
    train_dataset=dataset,
    task="churn",
    problem_type="binary",
)
print(trainer.train())

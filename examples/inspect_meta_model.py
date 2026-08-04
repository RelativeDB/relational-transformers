"""Inspect a Hub checkpoint without allocating its model weights."""

from relational_transformers import RelationalTransformer

model = RelationalTransformer("RelativeDB/rt-j-fp16", backend="meta")
dimensions = model.get_model_kwargs()
print("checkpoint:", model.model_name_or_path)
for name, value in dimensions.items():
    print(f"{name}: {value}")
print("parameter device:", next(model.model.parameters()).device)

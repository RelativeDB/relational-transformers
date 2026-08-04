"""Inspect a Hub checkpoint without allocating its model weights."""

from relational_transformers import RelationalTransformer

model = RelationalTransformer("RelativeDB/rt-j-fp16", backend="meta")
dimensions = model.get_model_kwargs()
assert dimensions == {
    "num_blocks": 12,
    "d_model": 512,
    "d_text": 384,
    "num_heads": 8,
    "d_ff": 2048,
}
assert next(model.model.parameters()).device.type == "meta"
print("checkpoint:", model.model_name_or_path)
for name, value in dimensions.items():
    print(f"{name}: {value}")
print("parameter device:", next(model.model.parameters()).device)

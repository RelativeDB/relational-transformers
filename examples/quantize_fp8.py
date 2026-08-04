"""Create a native E4M3 FP8 copy of a local or Hub checkpoint."""

from relational_transformers.quantization import quantize_model_fp8

output = quantize_model_fp8("RelativeDB/rt-j-fp16", "models/rt-j-fp8")
print(f"wrote classification and regression checkpoints to {output}")

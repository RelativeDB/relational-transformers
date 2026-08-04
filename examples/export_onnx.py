"""Export dynamic ONNX inference and compare it with PyTorch."""

from pathlib import Path

import numpy as np
from _common import issue_cells
from sentence_transformers import SentenceTransformer

from relational_transformers import RelationalTransformer

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
cells = issue_cells(
    encoder,
    title="Incorrect total after applying two coupons",
    body="Checkout displays a negative subtotal when discounts are stacked.",
)
torch_model = RelationalTransformer("RelativeDB/rt-j-fp16", backend="torch", device="cpu")
path = torch_model.export_onnx(Path("models/rt-j.onnx"), cells, target=0)
onnx_model = RelationalTransformer(path, backend="onnx")

expected = torch_model.predict(cells, target=0, activation="identity")
actual = onnx_model.predict(cells, target=0, activation="identity")
np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
print(f"exported {path}: score={actual:.4f}")

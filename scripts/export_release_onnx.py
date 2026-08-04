"""Export and verify the published RT-J checkpoint for release assets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from relational_transformers import DEFAULT_MODEL, RelationalTransformer


def example_context(*, cells: int, seed: int) -> np.ndarray:
    """Create deterministic model-ready cell vectors without an encoder dependency."""

    rng = np.random.default_rng(seed)
    values = rng.normal(size=(cells, 768)).astype(np.float32)
    values[0, 384:] = 0.0  # masked target value channel
    return values


def export_and_verify(
    output: str | Path,
    *,
    model_name: str = DEFAULT_MODEL,
) -> Path:
    """Export a real checkpoint and verify dynamic ONNX Runtime parity."""

    output = Path(output)
    torch_model = RelationalTransformer(model_name, backend="torch", device="cpu")
    export_input = example_context(cells=4, seed=7)
    torch_model.export_onnx(output, export_input, target=0)
    onnx_model = RelationalTransformer(output, backend="onnx")

    cases = [export_input, example_context(cells=7, seed=19)]
    expected = torch_model.predict(cases, target=0, activation="identity")
    actual = onnx_model.predict(cases, target=0, activation="identity")
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
    if not np.isfinite(actual).all():
        raise AssertionError("ONNX Runtime returned a non-finite score")

    max_error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    size_mib = output.stat().st_size / 1024**2
    print(f"verified {output} ({size_mib:.1f} MiB, max error {max_error:.3g})")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="dist/rt-j-classification.onnx")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    export_and_verify(args.output, model_name=args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

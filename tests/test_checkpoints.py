import pytest
import torch
from safetensors.torch import save_file

from relational_transformers.checkpoints import load_state


def test_load_state_expands_q4_nibbles(tmp_path):
    path = tmp_path / "model.q4.safetensors"
    packed = torch.tensor([[0x10, 0x32] * 8], dtype=torch.uint8)
    params = torch.tensor([[2.0, -1.0]], dtype=torch.float16)
    save_file({"projection.weight": packed, "projection.weight.q4_scale": params}, path)

    weight = load_state(path)["projection.weight"]

    assert weight.shape == (1, 32)
    assert weight[0, :4].tolist() == [-1.0, 1.0, 3.0, 5.0]


def test_load_state_expands_per_output_channel_q8(tmp_path):
    path = tmp_path / "model.q8.safetensors"
    quantized = torch.tensor([[2, -4, 6], [-3, 2, 1]], dtype=torch.int8)
    scales = torch.tensor([0.5, 2.0], dtype=torch.float32)
    save_file(
        {"projection.weight": quantized, "projection.weight.q_scale": scales},
        path,
    )

    weight = load_state(path)["projection.weight"]

    torch.testing.assert_close(
        weight,
        torch.tensor([[1.0, -2.0, 3.0], [-6.0, 4.0, 2.0]]),
    )


def test_load_state_rejects_malformed_q4_scales(tmp_path):
    path = tmp_path / "broken.q4.safetensors"
    save_file(
        {
            "projection.weight": torch.zeros((1, 16), dtype=torch.uint8),
            "projection.weight.q4_scale": torch.ones((1, 4), dtype=torch.float16),
        },
        path,
    )

    with pytest.raises(ValueError, match="invalid Q4 scales"):
        load_state(path)

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

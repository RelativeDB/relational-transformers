"""Local and Hugging Face checkpoint resolution."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

MODEL_KEYS = ("num_blocks", "d_model", "d_text", "num_heads", "d_ff")


def resolve_config(spec: str | Path, *, task="classification", revision=None) -> dict:
    path = Path(spec).expanduser()
    subfolder = "regression" if task in ("regression", "forecasting", "reg") else "classification"
    if path.is_file():
        config_path = path.with_name("config.json")
    elif path.is_dir():
        directory = path / subfolder if (path / subfolder).is_dir() else path
        config_path = directory / "config.json"
    else:
        from huggingface_hub import hf_hub_download

        config_path = Path(
            hf_hub_download(str(spec), f"{subfolder}/config.json", revision=revision)
        )
    return json.loads(config_path.read_text()) if config_path.exists() else {}


def resolve_checkpoint(
    spec: str | Path, *, task="classification", revision=None
) -> tuple[dict, Path]:
    path = Path(spec).expanduser()
    subfolder = "regression" if task in ("regression", "forecasting", "reg") else "classification"
    if path.is_file():
        config_path = path.with_name("config.json")
        return (json.loads(config_path.read_text()) if config_path.exists() else {}, path)
    if path.is_dir():
        directory = path / subfolder if (path / subfolder).is_dir() else path
        config = json.loads((directory / "config.json").read_text())
        return config, directory / config.get("checkpoint_file", "model.safetensors")
    from huggingface_hub import hf_hub_download

    config = resolve_config(spec, task=task, revision=revision)
    weights = Path(
        hf_hub_download(
            str(spec),
            f"{subfolder}/{config.get('checkpoint_file', 'model.safetensors')}",
            revision=revision,
        )
    )
    return config, weights


def model_dimensions(config: dict) -> dict:
    merged = {**{k: config[k] for k in MODEL_KEYS if k in config}, **config.get("model", {})}
    missing = [key for key in MODEL_KEYS if key not in merged]
    if missing:
        raise ValueError(f"checkpoint config is missing {missing}")
    return {key: merged[key] for key in MODEL_KEYS}


def load_state(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".pt":
        state = torch.load(path, map_location="cpu")
        state = state.get("model", state)
    else:
        state = load_file(str(path), device="cpu")
    float8_types = tuple(
        dtype
        for dtype in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e5m2", None),
        )
        if dtype is not None
    )
    state = {
        key: value.float() if value.dtype in float8_types else value for key, value in state.items()
    }
    adapted = {}
    for key, value in state.items():
        norm = (
            "norms." in key
            or "norm_dict." in key
            or key.startswith("norm_out.")
            or ".q_norm." in key
            or ".k_norm." in key
        )
        if key.endswith(".weight") and norm:
            key = key[: -len(".weight")] + ".scale"
        adapted[key] = value
    # Q8 checkpoints store an int tensor plus one scale per output row.
    for key in list(adapted):
        if not key.endswith(".q_scale"):
            continue
        weight_key = key[: -len(".q_scale")]
        if weight_key in adapted:
            scale = adapted.pop(key).float()
            weight = adapted[weight_key]
            adapted[weight_key] = weight.float() * scale.reshape((-1,) + (1,) * (weight.ndim - 1))
    # Q4 checkpoints pack adjacent values into the low/high nibbles of each
    # byte. Every group of 32 logical values stores an fp16 (scale, minimum)
    # pair, so portable PyTorch expands the packed representation at load.
    for key in list(adapted):
        if not key.endswith(".q4_scale"):
            continue
        weight_key = key[: -len(".q4_scale")]
        if weight_key not in adapted:
            continue
        packed = adapted[weight_key]
        if packed.ndim != 2 or packed.dtype != torch.uint8:
            raise ValueError(f"invalid Q4 payload for {weight_key}")
        rows, packed_width = packed.shape
        params = adapted.pop(key).float().reshape(rows, -1, 2)
        groups = params.shape[1]
        if packed_width != groups * 16:
            raise ValueError(f"invalid Q4 scales for {weight_key}")
        nibbles = torch.stack((packed & 0x0F, packed >> 4), dim=-1)
        nibbles = nibbles.reshape(rows, groups, 32).float()
        scale = params[..., 0].unsqueeze(-1)
        minimum = params[..., 1].unsqueeze(-1)
        adapted[weight_key] = (nibbles * scale + minimum).reshape(rows, groups * 32)
    return adapted


def save_checkpoint(model, directory: str | Path, config: dict) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        str(directory / "model.safetensors"),
    )
    config = {**config, "checkpoint_file": "model.safetensors"}
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")

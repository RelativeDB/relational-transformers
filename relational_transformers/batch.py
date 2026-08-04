"""The model-facing RT-J tensor contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .constants import BATCH_FIELDS, MAX_F2P, SEM_TEXT


@dataclass
class RelationalBatch:
    """A padded batch of relational cells.

    This is the shared boundary used by PyTorch, Triton, ONNX, and RelativeDB.
    Integer tensors describe topology; value tensors contain already encoded
    model inputs. Boolean masks use ``True`` for padding/targets.
    """

    node_idxs: Tensor
    f2p_nbr_idxs: Tensor
    col_name_idxs: Tensor
    table_name_idxs: Tensor
    is_padding: Tensor
    sem_types: Tensor
    is_targets: Tensor
    number_values: Tensor
    datetime_values: Tensor
    boolean_values: Tensor
    text_values: Tensor
    col_name_values: Tensor

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, Tensor):
                setattr(self, field.name, torch.as_tensor(value))
        self._canonicalize()
        self.validate()

    def _canonicalize(self) -> None:
        for name in ("node_idxs", "f2p_nbr_idxs", "col_name_idxs", "table_name_idxs", "sem_types"):
            setattr(self, name, getattr(self, name).to(torch.int64))
        self.is_padding = self.is_padding.bool()
        self.is_targets = self.is_targets.bool()
        for name in (
            "number_values",
            "datetime_values",
            "boolean_values",
            "text_values",
            "col_name_values",
        ):
            value = getattr(self, name)
            if name in ("number_values", "datetime_values", "boolean_values") and value.ndim == 2:
                value = value.unsqueeze(-1)
            if not value.is_floating_point():
                value = value.float()
            setattr(self, name, value)

    def validate(self) -> RelationalBatch:
        """Check shapes, semantic type range, target placement, and finiteness.

        Runs automatically on construction. Raises :class:`ValueError` with
        the offending field name on the first violation.
        """
        if self.node_idxs.ndim != 2:
            raise ValueError("node_idxs must have shape [batch, cells]")
        b, s = self.node_idxs.shape
        for name in ("col_name_idxs", "table_name_idxs", "is_padding", "sem_types", "is_targets"):
            if getattr(self, name).shape != (b, s):
                raise ValueError(f"{name} must have shape {(b, s)}")
        if self.f2p_nbr_idxs.shape != (b, s, MAX_F2P):
            raise ValueError(f"f2p_nbr_idxs must have shape {(b, s, MAX_F2P)}")
        for name in ("number_values", "datetime_values", "boolean_values"):
            if getattr(self, name).shape != (b, s, 1):
                raise ValueError(f"{name} must have shape {(b, s, 1)}")
        if self.text_values.ndim != 3 or self.text_values.shape[:2] != (b, s):
            raise ValueError("text_values must have shape [batch, cells, d_text]")
        if self.col_name_values.shape != self.text_values.shape:
            raise ValueError("col_name_values must match text_values")
        if ((self.sem_types < 0) | (self.sem_types > 3)).any():
            raise ValueError("sem_types values must be in [0, 3]")
        if (self.is_targets & self.is_padding).any():
            raise ValueError("a padding cell cannot be a target")
        if any(
            not torch.isfinite(getattr(self, name)).all()
            for name in ("number_values", "datetime_values", "boolean_values")
        ):
            raise ValueError("scalar channels must be finite")
        if (
            not torch.isfinite(self.text_values).all()
            or not torch.isfinite(self.col_name_values).all()
        ):
            raise ValueError("embedding channels must be finite")
        return self

    @property
    def batch_size(self) -> int:
        return self.node_idxs.shape[0]

    @property
    def sequence_length(self) -> int:
        return self.node_idxs.shape[1]

    @property
    def d_text(self) -> int:
        return self.text_values.shape[-1]

    def to(self, device: str | torch.device, dtype: torch.dtype | None = None) -> RelationalBatch:
        """Return a copy on ``device``, optionally casting floating-point channels."""
        values = {}
        for name in BATCH_FIELDS:
            value = getattr(self, name).to(device)
            if dtype is not None and value.is_floating_point():
                value = value.to(dtype)
            values[name] = value
        return RelationalBatch(**values)

    def as_dict(self) -> dict[str, Tensor]:
        """Return the canonical fields as a name-to-tensor dictionary.

        The keys match :data:`~relational_transformers.constants.BATCH_FIELDS`
        and round-trip through :meth:`from_mapping`.
        """
        return {name: getattr(self, name) for name in BATCH_FIELDS}

    def numpy(self) -> dict[str, np.ndarray]:
        """Return every canonical field as a CPU numpy array."""
        values = {}
        for name in BATCH_FIELDS:
            tensor = getattr(self, name).detach().cpu()
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            values[name] = tensor.numpy()
        return values

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> RelationalBatch:
        """Build a batch from a dictionary of arrays or tensors.

        Accepts numpy arrays, torch tensors, or nested lists for every field.
        Integer index fields are cast to ``int64``, ``is_padding`` and
        ``is_targets`` to ``bool`` (``uint8`` inputs work), and value channels
        to floating point. Short RelativeDB aliases such as ``f2p``,
        ``col_idxs``, ``table_idxs``, ``is_target``, ``number_v``, and
        ``text_v`` normalize to their canonical names. Raises
        :class:`ValueError` when a canonical field is missing.
        """
        aliases = {
            "f2p": "f2p_nbr_idxs",
            "col_idxs": "col_name_idxs",
            "table_idxs": "table_name_idxs",
            "is_target": "is_targets",
            "number_v": "number_values",
            "datetime_v": "datetime_values",
            "boolean_v": "boolean_values",
            "text_v": "text_values",
            "col_name_v": "col_name_values",
        }
        normalized = {aliases.get(name, name): value for name, value in values.items()}
        missing = [name for name in BATCH_FIELDS if name not in normalized]
        if missing:
            raise ValueError(f"relational batch is missing fields: {', '.join(missing)}")
        return cls(**{name: normalized[name] for name in BATCH_FIELDS})

    @classmethod
    def from_text_cells(
        cls,
        cells: np.ndarray | Tensor,
        *,
        target: int | Sequence[int],
        node_idxs: Sequence[int] | None = None,
        parents: Mapping[int, Sequence[int]] | None = None,
        table_idxs: Sequence[int] | None = None,
    ) -> RelationalBatch:
        """Build one batch row from ``[column_embedding, value_embedding]`` cells.

        ``cells`` has shape ``[S, 2*d_text]``. This convenience path is useful
        for all-text examples; typed production callers should construct the
        full batch so scalar and datetime channels retain their semantics.
        """

        cells = torch.as_tensor(cells, dtype=torch.float32)
        if cells.ndim != 2 or cells.shape[1] % 2:
            raise ValueError("text cells must have shape [cells, 2*d_text]")
        s, width = cells.shape
        d_text = width // 2
        targets = [target] if isinstance(target, int) else list(target)
        nodes = torch.as_tensor(node_idxs if node_idxs is not None else [0] * s).view(1, s)
        f2p = torch.full((1, s, MAX_F2P), -1, dtype=torch.long)
        for cell, values in (parents or {}).items():
            values = list(values)
            if len(values) > MAX_F2P:
                raise ValueError(f"cell {cell} has more than {MAX_F2P} parents")
            f2p[0, cell, : len(values)] = torch.tensor(values)
        target_mask = torch.zeros((1, s), dtype=torch.bool)
        target_mask[0, targets] = True
        zeros = torch.zeros((1, s, 1), dtype=torch.float32)
        return cls(
            node_idxs=nodes,
            f2p_nbr_idxs=f2p,
            col_name_idxs=torch.arange(s).view(1, s),
            table_name_idxs=torch.as_tensor(table_idxs if table_idxs is not None else [0] * s).view(
                1, s
            ),
            is_padding=torch.zeros((1, s), dtype=torch.bool),
            sem_types=torch.full((1, s), SEM_TEXT, dtype=torch.long),
            is_targets=target_mask,
            number_values=zeros.clone(),
            datetime_values=zeros.clone(),
            boolean_values=zeros.clone(),
            text_values=cells[:, d_text:].view(1, s, d_text),
            col_name_values=cells[:, :d_text].view(1, s, d_text),
        )

    def ablate(self, cells: Sequence[int]) -> RelationalBatch:
        """Return a new batch with the same cell positions padded out.

        Positions remain stable, so node and parent indices require no remap.
        Target cells may not be ablated.
        """

        positions = list(cells)
        if self.is_targets[:, positions].any():
            raise ValueError("target cells cannot be ablated")
        values = self.as_dict()
        values["is_padding"] = self.is_padding.clone()
        values["is_padding"][:, positions] = True
        return RelationalBatch(**values)

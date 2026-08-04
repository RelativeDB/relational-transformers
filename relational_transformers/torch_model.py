"""Portable dense PyTorch implementation of RT-J.

The parameter names intentionally match the published checkpoints and the
reference implementation in ``stanford-star/rt-j``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .batch import RelationalBatch

SEMANTICS = ("number", "text", "datetime", "boolean")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, *, device=None) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim, device=device))

    def forward(self, values: Tensor) -> Tensor:
        source = values.float()
        normalized = source * torch.rsqrt(source.square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(values.dtype) * self.scale


class MaskedAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, legacy_attn: bool = False, *, device=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.legacy_attn = legacy_attn
        self.q_norm = RMSNorm(self.head_dim, device=device)
        self.k_norm = RMSNorm(self.head_dim, device=device)
        self.wq = nn.Linear(d_model, d_model, bias=False, device=device)
        self.wk = nn.Linear(d_model, d_model, bias=False, device=device)
        self.wv = nn.Linear(d_model, d_model, bias=False, device=device)
        self.wo = nn.Linear(d_model, d_model, bias=False, device=device)
        if not legacy_attn:
            self.scale = nn.Parameter(torch.ones(1, num_heads, 1, 1, device=device))
            self.wg = nn.Linear(d_model, d_model, bias=False, device=device)

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        b, s, d = values.shape
        shape = (b, s, self.num_heads, self.head_dim)
        query = self.q_norm(self.wq(values).view(shape).permute(0, 2, 1, 3))
        key = self.k_norm(self.wk(values).view(shape).permute(0, 2, 1, 3))
        value = self.wv(values).view(shape).permute(0, 2, 1, 3).to(query.dtype)
        if not self.legacy_attn:
            counts = mask.sum(-1, keepdim=True).clamp_min(1).to(query.dtype)
            query = query * self.scale * counts.log().unsqueeze(1)
            scale = 1.0 / self.head_dim
        else:
            scale = self.head_dim**-0.5
        scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        valid = mask.unsqueeze(1)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = torch.where(valid.any(-1, keepdim=True), attention, torch.zeros_like(attention))
        output = torch.matmul(attention, value).permute(0, 2, 1, 3).reshape(b, s, d)
        if not self.legacy_attn:
            output = 2 * torch.sigmoid(self.wg(values)) * output
        return self.wo(output)


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, *, device=None):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False, device=device)
        self.w2 = nn.Linear(d_ff, d_model, bias=False, device=device)
        self.w3 = nn.Linear(d_model, d_ff, bias=False, device=device)

    def forward(self, values: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(values)) * self.w3(values))


class RelationalBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, legacy_attn=False, *, device=None):
        super().__init__()
        self.attn_types = ("col", "feat", "nbr")
        self.norms = nn.ModuleDict(
            {name: RMSNorm(d_model, device=device) for name in (*self.attn_types, "ffn")}
        )
        self.attns = nn.ModuleDict(
            {
                name: MaskedAttention(d_model, num_heads, legacy_attn, device=device)
                for name in self.attn_types
            }
        )
        self.ffn = FFN(d_model, d_ff, device=device)

    def forward(self, values: Tensor, masks: Mapping[str, Tensor]) -> Tensor:
        for name in self.attn_types:
            values = values + self.attns[name](self.norms[name](values), masks[name])
        return values + self.ffn(self.norms["ffn"](values))


@dataclass
class ModelOutput:
    scores: Tensor | None = None
    token_scores: Tensor | None = None
    target_text: Tensor | None = None
    features: Tensor | None = None
    embeddings: Tensor | None = None


class RTJModel(nn.Module):
    """The RT-J backbone and published decoder heads."""

    def __init__(
        self,
        num_blocks: int = 12,
        d_model: int = 512,
        d_text: int = 384,
        num_heads: int = 8,
        d_ff: int = 2048,
        legacy_attn: bool = False,
        *,
        device=None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_text = d_text
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.enc_dict = nn.ModuleDict(
            {
                "number": nn.Linear(1, d_model, device=device),
                "text": nn.Linear(d_text, d_model, device=device),
                "datetime": nn.Linear(1, d_model, device=device),
                "col_name": nn.Linear(d_text, d_model, device=device),
                "boolean": nn.Linear(1, d_model, device=device),
            }
        )
        self.dec_dict = nn.ModuleDict(
            {
                "number": nn.Linear(d_model, 1, device=device),
                "text": nn.Linear(d_model, d_text, device=device),
                "datetime": nn.Linear(d_model, 1, device=device),
                "boolean": nn.Linear(d_model, 1, device=device),
            }
        )
        self.norm_dict = nn.ModuleDict(
            {name: RMSNorm(d_model, device=device) for name in (*SEMANTICS, "col_name")}
        )
        self.mask_embs = nn.ParameterDict(
            {name: nn.Parameter(torch.empty(d_model, device=device)) for name in SEMANTICS}
        )
        self.blocks = nn.ModuleList(
            [
                RelationalBlock(d_model, num_heads, d_ff, legacy_attn, device=device)
                for _ in range(num_blocks)
            ]
        )
        self.norm_out = RMSNorm(d_model, device=device)

    def _masks(self, batch: RelationalBatch) -> dict[str, Tensor]:
        node = batch.node_idxs
        parents = batch.f2p_nbr_idxs
        valid = ~batch.is_padding
        pad = valid[:, :, None] & valid[:, None, :]
        same_node = node[:, :, None] == node[:, None, :]
        key_is_parent = (node[:, None, :, None] == parents[:, :, None, :]).any(-1)
        query_is_parent = (node[:, :, None, None] == parents[:, None, :, :]).any(-1)
        same_column = batch.col_name_idxs[:, :, None] == batch.col_name_idxs[:, None, :]
        same_table = batch.table_name_idxs[:, :, None] == batch.table_name_idxs[:, None, :]
        return {
            "feat": (same_node | key_is_parent) & pad,
            "nbr": query_is_parent & pad,
            "col": same_column & same_table & pad,
        }

    def encode(self, batch: RelationalBatch) -> Tensor:
        parameter = next(self.parameters())
        dtype = parameter.dtype
        masks = self._masks(batch)
        valid = ~batch.is_padding
        col_values = batch.col_name_values.to(dtype)
        values = self.norm_dict["col_name"](self.enc_dict["col_name"](col_values))
        values = values * valid.unsqueeze(-1)
        channels = {
            "number": batch.number_values,
            "text": batch.text_values,
            "datetime": batch.datetime_values,
            "boolean": batch.boolean_values,
        }
        for semantic, name in enumerate(SEMANTICS):
            channel = torch.nan_to_num(channels[name].to(dtype))
            present = (batch.sem_types == semantic) & ~batch.is_targets & valid
            target = (batch.sem_types == semantic) & batch.is_targets & valid
            encoded = self.norm_dict[name](self.enc_dict[name](channel))
            values = values + encoded * present.unsqueeze(-1)
            values = values + self.mask_embs[name] * target.unsqueeze(-1)
        for block in self.blocks:
            values = block(values, masks)
        return self.norm_out(values)

    def forward(self, batch: RelationalBatch, output: str = "target_scores") -> ModelOutput:
        embeddings = self.encode(batch)
        targets = batch.is_targets.to(embeddings.dtype)
        if output == "embeddings":
            return ModelOutput(embeddings=embeddings)
        if output == "target_features":
            return ModelOutput(features=(embeddings * targets.unsqueeze(-1)).sum(1))
        number = self.dec_dict["number"](embeddings).squeeze(-1)
        if output == "token_scores":
            return ModelOutput(token_scores=number)
        scores = (number * targets).sum(1)
        if output == "target_scores":
            return ModelOutput(scores=scores)
        if output == "target_scores_and_text":
            text = self.dec_dict["text"](embeddings)
            text = (text * targets.unsqueeze(-1)).sum(1)
            return ModelOutput(scores=scores, target_text=text)
        raise ValueError(f"unknown output {output!r}")

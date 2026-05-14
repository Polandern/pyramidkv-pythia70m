from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch


@dataclass
class CacheStats:
    """Small summary of retained KV-cache lengths."""

    layer_lengths: list[int]

    @property
    def average_length(self) -> float:
        if not self.layer_lengths:
            return 0.0
        return float(sum(self.layer_lengths) / len(self.layer_lengths))

    @property
    def total_tokens(self) -> int:
        return int(sum(self.layer_lengths))


def build_layer_budgets(
    num_layers: int,
    kv_budget: int,
    mode: str = "pyramid",
    min_budget: int = 32,
) -> list[int]:
    if mode == "uniform":
        return [kv_budget for _ in range(num_layers)]

    if mode != "pyramid":
        raise ValueError(f"Unknown budget mode: {mode}")

    # KVPress / PyramidKV-aligned schedule:
    # lower / earlier layers keep more KV cache,
    # higher / later layers keep less KV cache.
    high = max(min_budget, kv_budget)
    low = max(min_budget, kv_budget // 2)
    low = min(low, high)

    budgets = []
    for layer_idx in range(num_layers):
        ratio = layer_idx / max(1, num_layers - 1)
        budget = int(round(high - ratio * (high - low)))
        budget = max(1, budget)
        budgets.append(budget)

    return budgets


def compress_past_key_values(
    past_key_values,
    attentions: Optional[Iterable[torch.Tensor]],
    budgets: list[int],
    sink_tokens: int = 4,
    recent_tokens: int = 32,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Compress legacy tuple-style KV cache layer by layer.

    Each layer keeps a union of:
    1. first `sink_tokens`,
    2. most recent `recent_tokens`,
    3. high-attention tokens from the remaining middle region.

    This function intentionally operates on Hugging Face's legacy
    `past_key_values` tuples because they are stable and easy to inspect.
    """

    if past_key_values is None:
        return past_key_values

    attentions = list(attentions) if attentions is not None else []
    compressed = []

    for layer_idx, layer_past in enumerate(past_key_values):
        key, value = layer_past[:2]
        seq_len = key.shape[-2]
        budget = budgets[min(layer_idx, len(budgets) - 1)] if budgets else seq_len

        if seq_len <= budget:
            compressed.append((key, value))
            continue

        keep_idx = _select_keep_indices(
            seq_len=seq_len,
            budget=budget,
            sink_tokens=sink_tokens,
            recent_tokens=recent_tokens,
            attention=attentions[layer_idx] if layer_idx < len(attentions) else None,
            device=key.device,
        )
        compressed.append(
            (
                key.index_select(dim=-2, index=keep_idx),
                value.index_select(dim=-2, index=keep_idx),
            )
        )

    return tuple(compressed)


def cache_stats(past_key_values) -> CacheStats:
    if past_key_values is None:
        return CacheStats([])
    return CacheStats([int(layer[0].shape[-2]) for layer in past_key_values])


def _select_keep_indices(
    seq_len: int,
    budget: int,
    sink_tokens: int,
    recent_tokens: int,
    attention: Optional[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    budget = min(seq_len, max(1, int(budget)))
    sink_tokens = min(max(0, int(sink_tokens)), budget)
    recent_tokens = min(max(0, int(recent_tokens)), budget - sink_tokens)

    sink = torch.arange(0, sink_tokens, device=device, dtype=torch.long)
    recent_start = max(sink_tokens, seq_len - recent_tokens)
    recent = torch.arange(recent_start, seq_len, device=device, dtype=torch.long)

    remaining = budget - sink.numel() - recent.numel()
    if remaining <= 0:
        keep = torch.cat([sink, recent]).unique(sorted=True)
        return keep[-budget:] if keep.numel() > budget else keep

    middle_start = sink_tokens
    middle_end = recent_start
    if middle_end <= middle_start:
        keep = torch.cat([sink, recent]).unique(sorted=True)
        return keep[-budget:] if keep.numel() > budget else keep

    middle = torch.arange(middle_start, middle_end, device=device, dtype=torch.long)
    if attention is None:
        heavy = middle[-remaining:]
    else:
        scores = _attention_scores(attention, seq_len=seq_len, device=device)
        middle_scores = scores.index_select(dim=0, index=middle)
        topk = min(remaining, middle.numel())
        heavy = middle.index_select(dim=0, index=torch.topk(middle_scores, k=topk).indices)

    return torch.cat([sink, heavy, recent]).unique(sorted=True)


def _attention_scores(
    attention: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    # Expected shape: [batch, heads, query_len, key_len]. We use the latest
    # query token and average over batch/head dimensions.
    attn = attention.detach()
    while attn.dim() > 1:
        if attn.dim() == 4:
            attn = attn[:, :, -1, :]
            break
        attn = attn.mean(dim=0)
    scores = attn.mean(dim=tuple(range(attn.dim() - 1))) if attn.dim() > 1 else attn
    scores = scores.to(device=device, dtype=torch.float32)
    if scores.numel() < seq_len:
        padded = torch.zeros(seq_len, device=device, dtype=torch.float32)
        padded[-scores.numel() :] = scores
        return padded
    return scores[-seq_len:]


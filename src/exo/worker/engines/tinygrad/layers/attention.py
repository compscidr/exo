from __future__ import annotations

import math
from typing import cast

from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

from exo.worker.engines.tinygrad.cache import KVCache
from exo.worker.engines.tinygrad.layers.normalization import rms_norm
from exo.worker.engines.tinygrad.layers.rotary import apply_rope
from exo.worker.engines.tinygrad.quantization.layers import QuantizedLinear

LinearWeight = Tensor | QuantizedLinear

def linear_forward(x: Tensor, weight: LinearWeight) -> Tensor:
    if isinstance(weight, QuantizedLinear):
        return weight(x)
    return x @ weight.T

def grouped_query_attention(
    x: Tensor,
    qkv_proj: LinearWeight,
    o_proj: LinearWeight,
    cos_freqs: Tensor,
    sin_freqs: Tensor,
    cache: KVCache,
    layer_idx: int,
    position_offset: int | Tensor,
    cache_position: int | Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_norm: Tensor | None = None,
    k_norm: Tensor | None = None,
    qkv_bias: Tensor | None = None,
    o_bias: Tensor | None = None,
    rms_norm_eps: float = 1e-6,
) -> Tensor:
    _batch, seq_len, _ = x.shape

    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    qkv = linear_forward(x, qkv_proj)
    # Qwen2 (and some other architectures) have biased Q/K/V projections.
    # Llama has none, so qkv_bias is None and this branch is skipped.
    if qkv_bias is not None:
        qkv = qkv + qkv_bias
    q = qkv[..., :q_dim].reshape(int(_batch), seq_len, num_heads, head_dim).permute(0, 2, 1, 3)  # pyright: ignore[reportUnknownMemberType]
    k = qkv[..., q_dim:q_dim + kv_dim].reshape(int(_batch), seq_len, num_kv_heads, head_dim).permute(0, 2, 1, 3)  # pyright: ignore[reportUnknownMemberType]
    v = qkv[..., q_dim + kv_dim:].reshape(int(_batch), seq_len, num_kv_heads, head_dim).permute(0, 2, 1, 3)  # pyright: ignore[reportUnknownMemberType]

    if q_norm is not None:
        q = rms_norm(q, q_norm, rms_norm_eps)
    if k_norm is not None:
        k = rms_norm(k, k_norm, rms_norm_eps)

    q = apply_rope(q, cos_freqs, sin_freqs, position_offset)
    k = apply_rope(k, cos_freqs, sin_freqs, position_offset)

    # Store K,V in cache for future decode steps
    cache.update(layer_idx, k, v, position = cache_position)

    if isinstance(position_offset, int):
        # Prefill: compute attention against local K,V (seq_len × seq_len).
        # This avoids the wasteful (seq_len × max_seq_len) matmul that the
        # full-cache path would produce — up to 80× less work for short prompts.
        k_attn, v_attn = k, v
    else:
        # Decode (JIT): use full pre-allocated cache K,V.
        # Shapes are fixed (max_seq_len) which is required for TinyJit replay.
        k_attn = cache.keys[layer_idx]
        v_attn = cache.values[layer_idx]

    if num_kv_heads < num_heads:
        repeat_factor = num_heads // num_kv_heads
        k_attn = k_attn.unsqueeze(2).expand(  # pyright: ignore[reportUnknownMemberType]
            int(_batch), num_kv_heads, repeat_factor, -1, head_dim,
        ).reshape(int(_batch), num_heads, -1, head_dim)
        v_attn = v_attn.unsqueeze(2).expand(  # pyright: ignore[reportUnknownMemberType]
            int(_batch), num_kv_heads, repeat_factor, -1, head_dim,
        ).reshape(int(_batch), num_heads, -1, head_dim)

    scale = 1.0 / math.sqrt(head_dim)
    scores: Tensor = (q @ k_attn.transpose(-2, -1)) * scale

    if isinstance(position_offset, int):
        # Prefill from scratch: local K,V only (no prior cache), standard causal mask.
        if seq_len > 1:
            causal_mask = Tensor.ones(seq_len, seq_len).triu(1).reshape(1, 1, seq_len, seq_len)  # pyright: ignore[reportUnknownMemberType]
            scores = scores + causal_mask * float("-1e9")
    elif seq_len == 1:
        # Single-token decode: only the unfilled-positions mask needed (current row is the
        # only query position and is trivially causal against itself).
        valid_len = cache_position + seq_len  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
        col_indeces: Tensor = cache.col_indices
        unfilled_mask: Tensor = col_indeces >= valid_len  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
        scores = scores + unfilled_mask * float("-1e9")  # pyright: ignore[reportUnknownVariableType]
    else:
        # Batched incremental prefill: Tensor position_offset, seq_len > 1.
        # Each output row i (absolute position cache_position + i) may only
        # attend to key columns <= cache_position + i. col_indices is a
        # [max_seq_len] arange tensor the KVCache exposes. Broadcast against
        # per-row absolute positions to build a [1, 1, seq_len, max_seq_len]
        # disallow mask that also naturally masks unfilled positions (any
        # col > cache_position + seq_len - 1 is excluded).
        col_indeces_all: Tensor = cache.col_indices  # shape [1, 1, 1, max_seq_len]
        row_positions = cast(
            Tensor, cache_position
        ) + Tensor.arange(seq_len, dtype=dtypes.int32)  # pyright: ignore[reportUnknownMemberType] # shape [seq_len]
        row_positions = row_positions.reshape(1, 1, seq_len, 1)  # pyright: ignore[reportUnknownMemberType]
        col_indeces_row = col_indeces_all.reshape(1, 1, 1, -1)  # pyright: ignore[reportUnknownMemberType]
        disallow_mask: Tensor = col_indeces_row > row_positions
        scores = scores + disallow_mask * float("-1e9")

    attn_weights: Tensor = scores.softmax(axis=-1)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    out: Tensor = attn_weights @ v_attn  # pyright: ignore[reportUnknownVariableType]
    out = out.permute(0, 2, 1, 3).reshape(int(_batch), seq_len, num_heads * head_dim)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    out_projected = linear_forward(out, o_proj)  # pyright: ignore[reportUnknownArgumentType]
    if o_bias is not None:
        out_projected = out_projected + o_bias
    return out_projected

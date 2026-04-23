import numpy as np
import pytest
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

from exo.worker.engines.tinygrad.cache import KVCache
from exo.worker.engines.tinygrad.layers.attention import grouped_query_attention


@pytest.mark.slow
def test_batched_prefill_with_cache_matches_concatenated_prefill() -> None:
    """Prefilling [A, B] in one shot (from scratch) must populate the KV
    cache with the same contents as: prefill [A] from scratch, then batched-
    prefill [B] on top of the resulting cache (new combined-mask path)."""
    num_heads, num_kv_heads, head_dim = 2, 2, 8
    hidden = num_heads * head_dim
    max_seq_len = 32
    Tensor.manual_seed(42)

    def fresh_cache() -> KVCache:
        c = KVCache(
            num_layers=1, num_kv_heads=num_kv_heads,
            head_dim=head_dim, max_seq_len=max_seq_len,
        )
        c.keys[0] = c.keys[0].contiguous().realize()  # pyright: ignore[reportUnknownMemberType]
        c.values[0] = c.values[0].contiguous().realize()  # pyright: ignore[reportUnknownMemberType]
        return c

    qkv_proj = Tensor.rand(3 * hidden, hidden).realize()  # pyright: ignore[reportUnknownMemberType]
    o_proj = Tensor.rand(hidden, hidden).realize()  # pyright: ignore[reportUnknownMemberType]
    cos = Tensor.rand(max_seq_len, head_dim // 2).realize()  # pyright: ignore[reportUnknownMemberType]
    sin = Tensor.rand(max_seq_len, head_dim // 2).realize()  # pyright: ignore[reportUnknownMemberType]

    n_a, n_b = 4, 3
    x_a = Tensor.rand(1, n_a, hidden).realize()  # pyright: ignore[reportUnknownMemberType]
    x_b = Tensor.rand(1, n_b, hidden).realize()  # pyright: ignore[reportUnknownMemberType]
    x_full = x_a.cat(x_b, dim=1).contiguous().realize()  # pyright: ignore[reportUnknownMemberType]

    # Reference: single batched prefill covering [A, B] at position 0.
    cache_full = fresh_cache()
    _ = grouped_query_attention(
        x_full, qkv_proj=qkv_proj, o_proj=o_proj,
        cos_freqs=cos, sin_freqs=sin,
        cache=cache_full, layer_idx=0,
        position_offset=0, cache_position=0,
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
    )
    ref_keys: np.ndarray = cache_full.keys[0].numpy()
    ref_values: np.ndarray = cache_full.values[0].numpy()

    # Incremental: prefill A from scratch (int path), then batched incremental
    # prefill B with Tensor position_offset and seq_len=n_b (new path).
    cache_inc = fresh_cache()
    _ = grouped_query_attention(
        x_a, qkv_proj=qkv_proj, o_proj=o_proj,
        cos_freqs=cos, sin_freqs=sin,
        cache=cache_inc, layer_idx=0,
        position_offset=0, cache_position=0,
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
    )
    pos_tensor = Tensor([n_a], dtype=dtypes.int32).realize()
    _ = grouped_query_attention(
        x_b, qkv_proj=qkv_proj, o_proj=o_proj,
        cos_freqs=cos, sin_freqs=sin,
        cache=cache_inc, layer_idx=0,
        position_offset=pos_tensor, cache_position=pos_tensor,
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
    )
    inc_keys: np.ndarray = cache_inc.keys[0].numpy()
    inc_values: np.ndarray = cache_inc.values[0].numpy()

    # Cache positions 0..n_a+n_b-1 must match. Slots beyond that are undefined.
    np.testing.assert_allclose(ref_keys[:, :, :n_a + n_b, :], inc_keys[:, :, :n_a + n_b, :], atol=1e-4)
    np.testing.assert_allclose(ref_values[:, :, :n_a + n_b, :], inc_values[:, :, :n_a + n_b, :], atol=1e-4)

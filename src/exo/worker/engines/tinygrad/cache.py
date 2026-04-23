from typing import cast

from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor


class KVCache:
    def __init__(self, 
                 num_layers: int, 
                 num_kv_heads: int, 
                 head_dim: int, 
                 max_seq_len: int,
                 ) -> None:
        self.keys: list[Tensor] = [Tensor.zeros(1, num_kv_heads, max_seq_len, head_dim, dtype=dtypes.float16) for _ in range(num_layers)]  # pyright: ignore[reportUnknownMemberType]
        self.values: list[Tensor] = [Tensor.zeros(1, num_kv_heads, max_seq_len, head_dim, dtype=dtypes.float16) for _ in range(num_layers)]  # pyright: ignore[reportUnknownMemberType]

        self.max_seq_len = max_seq_len
        self._positions: Tensor = Tensor.arange(max_seq_len).reshape(1, 1, max_seq_len, 1)  # pyright: ignore[reportUnknownMemberType]
        self.col_indices: Tensor = Tensor.arange(max_seq_len).reshape(1, 1, 1, max_seq_len)  # pyright: ignore[reportUnknownMemberType]

    def update(
        self,
        layers_idx: int,
        key: Tensor,
        value: Tensor,
        position: int | Tensor = 0,
    ) -> tuple[Tensor, Tensor]:
        seq_len = key.shape[2]

        """
            Mask: For positions:
            [self.position, self.position + seq_len]

            We are using mask + pad since tinygrad tensor
            cannot handle slice assignment. 
        """

        positions = self._positions

        if isinstance(position, Tensor):
            # position is a scalar Tensor (shape [1]) giving the start slot.
            # For seq_len == 1 (single-token decode) or seq_len > 1 (batched
            # incremental prefill), write tokens into cache slots
            # [position, position + seq_len).
            #
            # Strategy: for each cache slot s in [0, max_seq_len), select the
            # corresponding source token index t = s - position if 0 <= t < seq_len,
            # else keep the existing cache value.
            #
            # Build slot_offset: (1, 1, max_seq_len, 1) − (1,) → (1, 1, max_seq_len, 1)
            all_slots = Tensor.arange(self.max_seq_len, dtype=dtypes.int32).reshape(1, 1, self.max_seq_len, 1)  # pyright: ignore[reportUnknownMemberType]
            slot_offset = all_slots - position
            in_range: Tensor = (slot_offset >= 0) & (slot_offset < seq_len)
            # Clamp offset so gather is in-bounds even for out-of-range slots
            safe_offset: Tensor = cast(Tensor, slot_offset.clip(0, seq_len - 1))  # pyright: ignore[reportUnknownMemberType]
            # key shape: (1, kv_heads, seq_len, head_dim)
            # Gather along seq dim: (1, kv_heads, max_seq_len, head_dim)
            gathered_k = key.half()[:, :, safe_offset.reshape(self.max_seq_len), :]  # pyright: ignore[reportUnknownMemberType]
            gathered_v = value.half()[:, :, safe_offset.reshape(self.max_seq_len), :]  # pyright: ignore[reportUnknownMemberType]
            self.keys[layers_idx] = Tensor.where(
                in_range, gathered_k, self.keys[layers_idx]
            )
            self.values[layers_idx] = Tensor.where(
                in_range, gathered_v, self.values[layers_idx]
            )
        else:
            pad_prev = position
            pad_next = self.max_seq_len - position - seq_len
            new_k = key.pad(
                ((0, 0), (0, 0), (pad_prev, pad_next), (0, 0))
            ).half()
            new_v = value.pad(
                ((0, 0), (0, 0), (pad_prev, pad_next), (0, 0))
            ).half()

            if position > 0:
                # Preserve existing cache slots outside [position, position+seq_len).
                # Must capture old cache tensors *before* any reassignment — an
                # earlier version that did `self.keys = new_k; if position > 0:
                # self.keys = where(mask, new_k, self.keys)` was trivially
                # self-referential (second RHS was the already-overwritten new_k)
                # and wiped the cache from position 0 to position.
                mask = (positions >= position) & (positions < position + seq_len)
                old_keys = self.keys[layers_idx]
                old_values = self.values[layers_idx]
                self.keys[layers_idx] = Tensor.where(mask, new_k, old_keys)
                self.values[layers_idx] = Tensor.where(mask, new_v, old_values)
            else:
                # Fresh prefill: padded new_k is the entire cache.
                self.keys[layers_idx] = new_k
                self.values[layers_idx] = new_v

        return self.keys[layers_idx], self.values[layers_idx]

    @property
    def seq_len(self) -> int:
        return int(self.keys[0].shape[2])

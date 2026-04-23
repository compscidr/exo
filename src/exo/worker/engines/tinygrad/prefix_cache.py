"""Per-model KV prefix cache registry for the pipeline generator.

Each pipeline rank keeps a module-level dict mapping ``id(model)`` to a
:class:`PrefixCacheState` holding the most recently populated ``KVCache``
and the tokens that produced it. On a new generation request, the
generator finds the longest common prefix of the new prompt and the
cached token sequence, then either extends the cache (incremental
prefill) or resets it (no useful overlap).
"""
from dataclasses import dataclass, field

from exo.worker.engines.tinygrad.cache import KVCache


@dataclass
class PrefixCacheState:
    """Per-model cache state on a single rank.

    ``cache``: the live KVCache; positions ``[0, next_position)`` are valid.

    ``tokens``: the token sequence that populated ``cache``. Only rank 0
    populates this meaningfully — worker ranks don't see raw prompt
    tokens (they receive hidden states over the wire) and leave it empty.

    ``next_position``: the next slot to write into — equal to the number
    of valid KV entries in ``cache``.
    """

    cache: KVCache
    tokens: list[int] = field(default_factory=list)
    next_position: int = 0


_registry: dict[int, PrefixCacheState] = {}


def prefix_cache_registry() -> dict[int, PrefixCacheState]:
    """Return the module-level registry. Exposed so tests can inspect/clear
    and so callers in the generator can do look-ups and writes directly.
    """
    return _registry


def find_common_prefix_length(cached: list[int], incoming: list[int]) -> int:
    """Return the length of the longest shared prefix of two token sequences.

    A full match of ``cached`` against the start of ``incoming`` returns
    ``len(cached)`` — signalling the cache covers a complete prefix of
    the new prompt.
    """
    n = min(len(cached), len(incoming))
    i = 0
    while i < n and cached[i] == incoming[i]:
        i += 1
    return i


def clear_prefix_cache(model_id: int) -> None:
    """Drop cached state for ``model_id``. Called when the caller decides
    the cached prefix isn't worth reusing."""
    _registry.pop(model_id, None)

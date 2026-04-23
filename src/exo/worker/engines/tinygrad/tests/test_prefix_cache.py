from exo.worker.engines.tinygrad.prefix_cache import (
    find_common_prefix_length,
    prefix_cache_registry,
)


def test_find_common_prefix_length_full_match() -> None:
    assert find_common_prefix_length([1, 2, 3], [1, 2, 3]) == 3


def test_find_common_prefix_length_partial_match() -> None:
    assert find_common_prefix_length([1, 2, 3, 4], [1, 2, 9, 8]) == 2


def test_find_common_prefix_length_no_match() -> None:
    assert find_common_prefix_length([1, 2, 3], [9, 8, 7]) == 0


def test_find_common_prefix_length_empty_inputs() -> None:
    assert find_common_prefix_length([], [1, 2, 3]) == 0
    assert find_common_prefix_length([1, 2, 3], []) == 0


def test_find_common_prefix_length_extension() -> None:
    """New prompt extends cached sequence — all cached tokens match."""
    assert find_common_prefix_length([1, 2, 3], [1, 2, 3, 4, 5]) == 3


def test_registry_set_get_clear() -> None:
    reg = prefix_cache_registry()
    key = 12345
    # Use a string as a stand-in for state in the pure-registry test; real
    # callers insert PrefixCacheState instances. The registry dict is
    # untyped-value on purpose (any Python object).
    reg[key] = "hello"  # pyright: ignore[reportArgumentType]
    assert reg[key] == "hello"
    reg.pop(key, None)
    assert key not in reg

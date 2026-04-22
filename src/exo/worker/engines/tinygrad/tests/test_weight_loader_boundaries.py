"""Tests for boundary-aware weight loading (pipeline parallel rank 0 / last / middle)."""

from __future__ import annotations

from pathlib import Path

# pyright: reportUnknownVariableType=false
import numpy as np
from safetensors.numpy import save_file

from exo.shared.architecture.llama import LLAMA_SPEC
from exo.shared.model_config import ModelConfig

# ── Tiny-model fixture helpers ─────────────────────────────────────────────

_HIDDEN = 32
_INTERMEDIATE = 64
_VOCAB = 128
_HEADS = 2
_KV_HEADS = 2
_NUM_LAYERS = 2


def _make_config(*, tie_word_embeddings: bool = False) -> ModelConfig:
    return ModelConfig(
        architecture_spec=LLAMA_SPEC,
        num_hidden_layers=_NUM_LAYERS,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        num_attention_heads=_HEADS,
        num_key_value_heads=_KV_HEADS,
        vocab_size=_VOCAB,
        head_dim=_HIDDEN // _HEADS,
        rope_theta=10000.0,
        rope_scaling=None,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        tie_word_embeddings=tie_word_embeddings,
        quantization_config=None,
    )


def _f32(*shape: int) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
    """Return a zero float32 numpy array of the given shape."""
    return np.zeros(shape, dtype=np.float32)


def _layer_arrays(layer_idx: int) -> dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float32]]]:
    """Return numpy arrays for one transformer layer (Llama key names)."""
    prefix = f"model.layers.{layer_idx}"
    head_dim = _HIDDEN // _HEADS
    return {
        f"{prefix}.self_attn.q_proj.weight": _f32(_HEADS * head_dim, _HIDDEN),
        f"{prefix}.self_attn.k_proj.weight": _f32(_KV_HEADS * head_dim, _HIDDEN),
        f"{prefix}.self_attn.v_proj.weight": _f32(_KV_HEADS * head_dim, _HIDDEN),
        f"{prefix}.self_attn.o_proj.weight": _f32(_HIDDEN, _HIDDEN),
        f"{prefix}.mlp.gate_proj.weight": _f32(_INTERMEDIATE, _HIDDEN),
        f"{prefix}.mlp.up_proj.weight": _f32(_INTERMEDIATE, _HIDDEN),
        f"{prefix}.mlp.down_proj.weight": _f32(_HIDDEN, _INTERMEDIATE),
        f"{prefix}.input_layernorm.weight": _f32(_HIDDEN),
        f"{prefix}.post_attention_layernorm.weight": _f32(_HIDDEN),
    }


def _build_fake_model(path: Path, *, with_lm_head: bool = True) -> None:
    """Write a minimal safetensors file for a 2-layer Llama-like model."""
    arrays: dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float32]]] = {
        "model.embed_tokens.weight": _f32(_VOCAB, _HIDDEN),
        "model.norm.weight": _f32(_HIDDEN),
    }
    if with_lm_head:
        arrays["lm_head.weight"] = _f32(_VOCAB, _HIDDEN)

    for i in range(_NUM_LAYERS):
        arrays.update(_layer_arrays(i))

    save_file(arrays, str(path / "model.safetensors"))


# ── Tests ──────────────────────────────────────────────────────────────────


def test_default_loads_everything(tmp_path: Path) -> None:
    """Calling with defaults (is_first_rank=True, is_last_rank=True) must
    populate embed_tokens, lm_head, and final_norm."""
    from exo.worker.engines.tinygrad.weight_loader import load_transformer_weights

    _build_fake_model(tmp_path)
    config = _make_config()
    weights = load_transformer_weights(tmp_path, config)

    assert weights.embed_tokens is not None
    assert weights.lm_head is not None
    assert weights.final_norm is not None
    assert len(weights.layers) == _NUM_LAYERS


def test_middle_rank_skips_boundary_weights(tmp_path: Path) -> None:
    """A middle pipeline rank (is_first_rank=False, is_last_rank=False) must
    have embed_tokens, lm_head, and final_norm all set to None."""
    from exo.worker.engines.tinygrad.weight_loader import load_transformer_weights

    _build_fake_model(tmp_path)
    config = _make_config()
    weights = load_transformer_weights(
        tmp_path, config, is_first_rank=False, is_last_rank=False
    )

    assert weights.embed_tokens is None
    assert weights.lm_head is None
    assert weights.final_norm is None
    assert len(weights.layers) == _NUM_LAYERS


def test_first_rank_has_embed_not_lm_head(tmp_path: Path) -> None:
    """Rank 0 (is_first_rank=True, is_last_rank=False) must have embed_tokens
    but not lm_head or final_norm."""
    from exo.worker.engines.tinygrad.weight_loader import load_transformer_weights

    _build_fake_model(tmp_path)
    config = _make_config()
    weights = load_transformer_weights(
        tmp_path, config, is_first_rank=True, is_last_rank=False
    )

    assert weights.embed_tokens is not None
    assert weights.lm_head is None
    assert weights.final_norm is None


def test_last_rank_has_lm_head_not_embed(tmp_path: Path) -> None:
    """Last rank (is_first_rank=False, is_last_rank=True) must have lm_head
    and final_norm but not embed_tokens."""
    from exo.worker.engines.tinygrad.weight_loader import load_transformer_weights

    _build_fake_model(tmp_path)
    config = _make_config()
    weights = load_transformer_weights(
        tmp_path, config, is_first_rank=False, is_last_rank=True
    )

    assert weights.embed_tokens is None
    assert weights.lm_head is not None
    assert weights.final_norm is not None


def test_last_rank_tied_embeddings(tmp_path: Path) -> None:
    """For a tied-embedding model on the last (non-first) rank:
    - lm_head must be populated (sourced from embed weights)
    - embed_tokens must remain None (we don't need to embed on this rank)
    """
    from exo.worker.engines.tinygrad.weight_loader import load_transformer_weights

    # Tied models don't have a separate lm_head.weight key — omit it.
    _build_fake_model(tmp_path, with_lm_head=False)
    config = _make_config(tie_word_embeddings=True)

    weights = load_transformer_weights(
        tmp_path, config, is_first_rank=False, is_last_rank=True
    )

    # embed_tokens must NOT be populated (last-only rank has no use for it)
    assert weights.embed_tokens is None
    # lm_head MUST be populated (sourced from the embed weight)
    assert weights.lm_head is not None
    assert weights.final_norm is not None


def test_single_rank_tied_embeddings_loads_both(tmp_path: Path) -> None:
    """For a tied-embedding model on a single (first+last) rank:
    both embed_tokens and lm_head must be populated."""
    from exo.worker.engines.tinygrad.weight_loader import load_transformer_weights

    _build_fake_model(tmp_path, with_lm_head=False)
    config = _make_config(tie_word_embeddings=True)

    weights = load_transformer_weights(
        tmp_path, config, is_first_rank=True, is_last_rank=True
    )

    assert weights.embed_tokens is not None
    assert weights.lm_head is not None
    assert weights.final_norm is not None

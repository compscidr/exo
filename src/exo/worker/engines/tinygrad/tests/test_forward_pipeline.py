"""Tests for pipeline-parallel forward_pass behaviour.

Verifies that forward_pass correctly handles all four rank configurations:
  - single-rank (embed + lm_head present): token IDs in, logits out
  - first-only rank (embed present, lm_head absent): token IDs in, hidden out
  - middle rank (neither embed nor lm_head): hidden in, hidden out
  - last-only rank (lm_head present, embed absent): hidden in, logits out

Numerical correctness is NOT checked; only output shapes and the ability to run
without crashing are verified.
"""

from __future__ import annotations

from pathlib import Path

# pyright: reportUnknownVariableType=false
import numpy as np
from safetensors.numpy import save_file
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

from exo.shared.architecture.llama import LLAMA_SPEC
from exo.shared.model_config import ModelConfig

# ── Tiny-model constants (same as test_weight_loader_boundaries) ──────────────

_HIDDEN = 32
_INTERMEDIATE = 64
_VOCAB = 128
_HEADS = 2
_KV_HEADS = 2
_NUM_LAYERS = 2
_HEAD_DIM = _HIDDEN // _HEADS


def _make_config() -> ModelConfig:
    return ModelConfig(
        architecture_spec=LLAMA_SPEC,
        num_hidden_layers=_NUM_LAYERS,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        num_attention_heads=_HEADS,
        num_key_value_heads=_KV_HEADS,
        vocab_size=_VOCAB,
        head_dim=_HEAD_DIM,
        rope_theta=10000.0,
        rope_scaling=None,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        quantization_config=None,
    )


def _f32(*shape: int) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
    """Return a zero float32 numpy array of the given shape."""
    return np.zeros(shape, dtype=np.float32)


def _layer_arrays(layer_idx: int) -> dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float32]]]:
    prefix = f"model.layers.{layer_idx}"
    return {
        f"{prefix}.self_attn.q_proj.weight": _f32(_HEADS * _HEAD_DIM, _HIDDEN),
        f"{prefix}.self_attn.k_proj.weight": _f32(_KV_HEADS * _HEAD_DIM, _HIDDEN),
        f"{prefix}.self_attn.v_proj.weight": _f32(_KV_HEADS * _HEAD_DIM, _HIDDEN),
        f"{prefix}.self_attn.o_proj.weight": _f32(_HIDDEN, _HIDDEN),
        f"{prefix}.mlp.gate_proj.weight": _f32(_INTERMEDIATE, _HIDDEN),
        f"{prefix}.mlp.up_proj.weight": _f32(_INTERMEDIATE, _HIDDEN),
        f"{prefix}.mlp.down_proj.weight": _f32(_HIDDEN, _INTERMEDIATE),
        f"{prefix}.input_layernorm.weight": _f32(_HIDDEN),
        f"{prefix}.post_attention_layernorm.weight": _f32(_HIDDEN),
    }


def _build_fake_model(path: Path) -> None:
    arrays: dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float32]]] = {
        "model.embed_tokens.weight": _f32(_VOCAB, _HIDDEN),
        "model.norm.weight": _f32(_HIDDEN),
        "lm_head.weight": _f32(_VOCAB, _HIDDEN),
    }
    for i in range(_NUM_LAYERS):
        arrays.update(_layer_arrays(i))
    save_file(arrays, str(path / "model.safetensors"))


# ── Helpers ───────────────────────────────────────────────────────────────────

_SEQ = 4  # short sequence length used in all tests


def _token_ids() -> Tensor:
    """Return a [1, _SEQ] int32 tensor of token IDs (all zeros, fine for shape tests)."""
    return Tensor.zeros(1, _SEQ, dtype=dtypes.int32)  # pyright: ignore[reportUnknownMemberType]


def _hidden_state() -> Tensor:
    """Return a [1, _SEQ, _HIDDEN] float32 hidden-state tensor."""
    return Tensor.zeros(1, _SEQ, _HIDDEN)  # pyright: ignore[reportUnknownMemberType]


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_full_pass_returns_logits(tmp_path: Path) -> None:
    """Single-rank (first + last): token IDs in → logits [1, seq, vocab] out."""
    from exo.worker.engines.tinygrad.forward import forward_pass
    from exo.worker.engines.tinygrad.weight_loader import load_transformer_weights

    _build_fake_model(tmp_path)
    config = _make_config()
    weights = load_transformer_weights(
        tmp_path, config, is_first_rank=True, is_last_rank=True
    )

    output, _ = forward_pass(weights, _token_ids(), None)
    output = output.realize()

    assert output.shape == (1, _SEQ, _VOCAB), (
        f"Expected logits shape (1, {_SEQ}, {_VOCAB}), got {output.shape}"
    )


def test_middle_rank_returns_hidden(tmp_path: Path) -> None:
    """Middle rank (no embed, no lm_head): hidden in → hidden [1, seq, hidden] out."""
    from exo.worker.engines.tinygrad.forward import forward_pass
    from exo.worker.engines.tinygrad.weight_loader import load_transformer_weights

    _build_fake_model(tmp_path)
    config = _make_config()
    weights = load_transformer_weights(
        tmp_path, config, is_first_rank=False, is_last_rank=False
    )

    assert weights.embed_tokens is None
    assert weights.lm_head is None

    output, _ = forward_pass(weights, _hidden_state(), None)
    output = output.realize()

    assert output.shape == (1, _SEQ, _HIDDEN), (
        f"Expected hidden shape (1, {_SEQ}, {_HIDDEN}), got {output.shape}"
    )


def test_first_rank_returns_hidden(tmp_path: Path) -> None:
    """First-only rank (embed present, lm_head absent): token IDs in → hidden out."""
    from exo.worker.engines.tinygrad.forward import forward_pass
    from exo.worker.engines.tinygrad.weight_loader import load_transformer_weights

    _build_fake_model(tmp_path)
    config = _make_config()
    weights = load_transformer_weights(
        tmp_path, config, is_first_rank=True, is_last_rank=False
    )

    assert weights.embed_tokens is not None
    assert weights.lm_head is None

    output, _ = forward_pass(weights, _token_ids(), None)
    output = output.realize()

    assert output.shape == (1, _SEQ, _HIDDEN), (
        f"Expected hidden shape (1, {_SEQ}, {_HIDDEN}), got {output.shape}"
    )


def test_last_rank_returns_logits_from_hidden(tmp_path: Path) -> None:
    """Last-only rank (lm_head present, embed absent): hidden in → logits out."""
    from exo.worker.engines.tinygrad.forward import forward_pass
    from exo.worker.engines.tinygrad.weight_loader import load_transformer_weights

    _build_fake_model(tmp_path)
    config = _make_config()
    weights = load_transformer_weights(
        tmp_path, config, is_first_rank=False, is_last_rank=True
    )

    assert weights.embed_tokens is None
    assert weights.lm_head is not None

    output, _ = forward_pass(weights, _hidden_state(), None)
    output = output.realize()

    assert output.shape == (1, _SEQ, _VOCAB), (
        f"Expected logits shape (1, {_SEQ}, {_VOCAB}), got {output.shape}"
    )

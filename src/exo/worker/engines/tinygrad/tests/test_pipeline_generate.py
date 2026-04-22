"""Tests for the pipeline-parallel tinygrad_generate dispatch (Task 6).

All tests use a tiny synthetic model (2 layers, 32-dim hidden, 128 vocab)
created in a tmp_path, with mocked PipelineGroup transports.  Numerical
correctness is NOT checked — only structural behaviour (correct calls,
correct shapes, correct ordering).
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save_file  # pyright: ignore[reportUnknownVariableType]

from exo.shared.architecture.llama import LLAMA_SPEC
from exo.shared.model_config import ModelConfig
from exo.shared.types.common import ModelId as CommonModelId
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.worker.engines.tinygrad.pipeline_group import (
    TAG_HIDDEN,
    TAG_STOP,
    TAG_TOKEN,
    PipelineGroup,
    decode_hidden,
    encode_hidden,
    encode_token,
)

# ── Tiny-model constants (mirrors test_weight_loader_boundaries) ───────────

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


def _build_fake_model(path: Path, *, with_lm_head: bool = True) -> None:
    arrays: dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float32]]] = {
        "model.embed_tokens.weight": _f32(_VOCAB, _HIDDEN),
        "model.norm.weight": _f32(_HIDDEN),
    }
    if with_lm_head:
        arrays["lm_head.weight"] = _f32(_VOCAB, _HIDDEN)
    for i in range(_NUM_LAYERS):
        arrays.update(_layer_arrays(i))
    save_file(arrays, str(path / "model.safetensors"))


# ── Minimal tokenizer stub ─────────────────────────────────────────────────

class _StubTokenizer:
    """Minimal tokenizer that encodes as [1, 2, 3] and decodes to a string."""

    eos_token_id: int = 2

    def encode(self, text: str) -> list[int]:
        _ = text
        return [1, 2, 3]

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)


# ── Task stub ──────────────────────────────────────────────────────────────

def _make_task() -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=CommonModelId("test-model"),
        input=[InputMessage(role="user", content="hello")],
        max_output_tokens=3,
    )


# ── Typed fake PipelineGroup ───────────────────────────────────────────────

class _FakeGroup(PipelineGroup):
    """Typed replacement for MagicMock that satisfies basedpyright's strict reportAny.

    Instead of open sockets it uses queued bytes-payloads for recv_any and
    records all outbound calls so tests can inspect them.
    """

    # Payloads queued by the test for the SUT to receive via recv_any().
    inbound_queue: list[tuple[int, bytes]]

    # Records of what the SUT sent to the "next" rank.
    sent_hiddens: list[np.ndarray[Any, np.dtype[np.float32]]]
    sent_tokens: list[tuple[int, bool]]
    sent_stops: int

    def __init__(self, *, rank: int, world_size: int) -> None:
        # Use a null socket as placeholder — we override all I/O methods below.
        null_sock = socket.socket()
        null_sock.close()
        # Call dataclass __init__ directly via object.__setattr__ to skip socket logic.
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "world_size", world_size)
        object.__setattr__(self, "recv_sock", null_sock)
        object.__setattr__(self, "send_sock", null_sock)
        self.inbound_queue = []
        self.sent_hiddens = []
        self.sent_tokens = []
        self.sent_stops = 0

    # ── Overridden transport methods ─────────────────────────────────────

    def send_hidden(self, arr: np.ndarray[Any, np.dtype[np.float32]]) -> None:
        self.sent_hiddens.append(arr)

    def recv_hidden(self) -> np.ndarray[Any, np.dtype[np.float32]]:
        tag, payload = self.recv_any()
        if tag != TAG_HIDDEN:
            raise RuntimeError(f"expected HIDDEN, got tag={tag}")
        return decode_hidden(payload)

    def send_token(self, token_id: int, stop: bool) -> None:
        self.sent_tokens.append((token_id, stop))

    def recv_token(self) -> tuple[int, bool]:
        tag, payload = self.recv_any()
        if tag != TAG_TOKEN:
            raise RuntimeError(f"expected TOKEN, got tag={tag}")
        from exo.worker.engines.tinygrad.pipeline_group import decode_token
        return decode_token(payload)

    def send_stop(self) -> None:
        self.sent_stops += 1

    def recv_any(self) -> tuple[int, bytes]:
        if not self.inbound_queue:
            raise RuntimeError("_FakeGroup.inbound_queue exhausted")
        return self.inbound_queue.pop(0)

    def close(self) -> None:
        pass

    # ── Helper to pre-load inbound messages ──────────────────────────────

    def queue_hidden(self, arr: np.ndarray[Any, np.dtype[np.float32]]) -> None:
        self.inbound_queue.append((TAG_HIDDEN, encode_hidden(arr)))

    def queue_stop(self) -> None:
        self.inbound_queue.append((TAG_STOP, b""))

    def queue_token(self, token_id: int, stop: bool = False) -> None:
        self.inbound_queue.append((TAG_TOKEN, encode_token(token_id, stop)))


def _fake_group(rank: int, world_size: int) -> _FakeGroup:
    return _FakeGroup(rank=rank, world_size=world_size)


# ── Tests ──────────────────────────────────────────────────────────────────

def test_single_rank_path_unchanged(tmp_path: Path) -> None:
    """group=None must fall through to _single_rank_generate without crashing.

    Uses a full model (embed + lm_head).  We consume up to 3 tokens.
    This guards the refactor from regression.
    """
    from exo.shared.types.worker.runner_response import GenerationResponse
    from exo.worker.engines.tinygrad.generator.generate import tinygrad_generate
    from exo.worker.engines.tinygrad.weight_loader import load_transformer_weights

    _build_fake_model(tmp_path)
    config = _make_config()
    weights = load_transformer_weights(tmp_path, config, is_first_rank=True, is_last_rank=True)

    tokenizer = _StubTokenizer()
    task = _make_task()
    prompt = "hello world"

    responses: list[GenerationResponse] = []
    for resp in tinygrad_generate(weights, tokenizer, task, prompt, group=None):
        responses.append(resp)
        if len(responses) >= 3:
            break

    assert len(responses) >= 1, "expected at least one GenerationResponse"
    for resp in responses:
        assert isinstance(resp, GenerationResponse)


def test_world_size_1_uses_single_rank_path(tmp_path: Path) -> None:
    """group.world_size == 1 must also use _single_rank_generate."""
    from exo.shared.types.worker.runner_response import GenerationResponse
    from exo.worker.engines.tinygrad.generator.generate import tinygrad_generate
    from exo.worker.engines.tinygrad.weight_loader import load_transformer_weights

    _build_fake_model(tmp_path)
    config = _make_config()
    weights = load_transformer_weights(tmp_path, config, is_first_rank=True, is_last_rank=True)

    tokenizer = _StubTokenizer()
    task = _make_task()

    group = _fake_group(rank=0, world_size=1)

    responses: list[GenerationResponse] = []
    for resp in tinygrad_generate(weights, tokenizer, task, "hello world", group=group):
        responses.append(resp)

    assert len(responses) >= 1
    # No pipeline transport calls should have been made.
    assert len(group.sent_hiddens) == 0
    assert len(group.sent_tokens) == 0


def test_pipeline_rank0_sends_hidden_recvs_token(tmp_path: Path) -> None:
    """Rank-0 pipeline path: send_hidden called with 3-D fp16 array;
    recv_token feeds tokens; GenerationResponse chunks yielded.
    """
    from exo.shared.types.worker.runner_response import GenerationResponse
    from exo.worker.engines.tinygrad.generator.generate import (
        _rank0_pipeline_generate,  # pyright: ignore[reportPrivateUsage]
    )
    from exo.worker.engines.tinygrad.weight_loader import load_transformer_weights

    _build_fake_model(tmp_path)
    config = _make_config()
    # Rank 0: has embed, no lm_head.
    weights = load_transformer_weights(tmp_path, config, is_first_rank=True, is_last_rank=False)

    assert weights.embed_tokens is not None
    assert weights.lm_head is None

    tokenizer = _StubTokenizer()
    task = _make_task()

    group = _fake_group(rank=0, world_size=2)
    # Simulate last rank returning two tokens then a stop-flagged token.
    group.queue_token(10, stop=False)
    group.queue_token(11, stop=False)
    group.queue_token(12, stop=True)

    responses: list[GenerationResponse] = list(
        _rank0_pipeline_generate(weights, tokenizer, task, "hello world", group)
    )

    # send_hidden must have been called at least once (prefill).
    assert len(group.sent_hiddens) >= 1, "expected at least one send_hidden call"

    # All send_hidden calls must pass a 3-D fp16 ndarray.
    for arr in group.sent_hiddens:
        arr_shape: tuple[int, ...] = arr.shape  # pyright: ignore[reportAny]
        assert arr.dtype == np.float32, f"expected fp32, got {arr.dtype}"
        assert len(arr_shape) == 3, f"expected 3-D hidden, got {len(arr_shape)}-D"

    # Must have yielded GenerationResponse objects.
    assert len(responses) >= 1
    for resp in responses:
        assert isinstance(resp, GenerationResponse)

    # send_stop must have been called to unblock workers.
    assert group.sent_stops >= 1


def test_rank0_pipeline_sends_stop_on_finish(tmp_path: Path) -> None:
    """Rank-0 must call send_stop when generation finishes naturally."""
    from exo.worker.engines.tinygrad.generator.generate import (
        _rank0_pipeline_generate,  # pyright: ignore[reportPrivateUsage]
    )
    from exo.worker.engines.tinygrad.weight_loader import load_transformer_weights

    _build_fake_model(tmp_path)
    config = _make_config()
    weights = load_transformer_weights(tmp_path, config, is_first_rank=True, is_last_rank=False)

    tokenizer = _StubTokenizer()
    # Tokenizer.eos_token_id = 2, so recv_token returning 2 triggers EOS.
    task = _make_task()

    group = _fake_group(rank=0, world_size=2)
    # Return EOS token immediately.
    group.queue_token(2, stop=False)

    responses = list(
        _rank0_pipeline_generate(weights, tokenizer, task, "hello", group)
    )

    # Should have sent stop.
    assert group.sent_stops >= 1
    # The EOS response should have finish_reason == "stop".
    assert any(r.finish_reason == "stop" for r in responses)


def test_worker_loop_processes_hidden_and_sends(tmp_path: Path) -> None:
    """Middle worker: recv HIDDEN -> forward -> send_hidden to next rank.
    Then recv STOP -> propagate send_stop -> return.
    """
    from exo.worker.engines.tinygrad.generator.generate import (
        _worker_pipeline_loop,  # pyright: ignore[reportPrivateUsage]
    )
    from exo.worker.engines.tinygrad.weight_loader import load_transformer_weights

    _build_fake_model(tmp_path)
    config = _make_config()
    # Middle rank: no embed, no lm_head.
    weights = load_transformer_weights(tmp_path, config, is_first_rank=False, is_last_rank=False)

    group = _fake_group(rank=1, world_size=3)
    # Queue a fake hidden then a stop.
    arr_in = np.zeros((1, 1, _HIDDEN), dtype=np.float32)
    group.queue_hidden(arr_in)
    group.queue_stop()

    _worker_pipeline_loop(weights, group, is_last=False)

    # send_hidden must have been called once with a 3-D fp16 ndarray.
    assert len(group.sent_hiddens) == 1
    sent = group.sent_hiddens[0]
    sent_shape: tuple[int, ...] = sent.shape  # pyright: ignore[reportAny]
    assert sent.dtype == np.float32
    assert len(sent_shape) == 3

    # STOP must have been propagated.
    assert group.sent_stops == 1


def test_last_rank_samples_and_sends_token(tmp_path: Path) -> None:
    """Last worker: recv HIDDEN -> forward -> send_token (not send_hidden).
    Then recv STOP -> propagate.
    """
    from exo.worker.engines.tinygrad.generator.generate import (
        _worker_pipeline_loop,  # pyright: ignore[reportPrivateUsage]
    )
    from exo.worker.engines.tinygrad.weight_loader import load_transformer_weights

    _build_fake_model(tmp_path)
    config = _make_config()
    # Last rank: no embed, has lm_head.
    weights = load_transformer_weights(tmp_path, config, is_first_rank=False, is_last_rank=True)

    assert weights.embed_tokens is None
    assert weights.lm_head is not None

    group = _fake_group(rank=1, world_size=2)
    arr_in = np.zeros((1, 1, _HIDDEN), dtype=np.float32)
    group.queue_hidden(arr_in)
    group.queue_stop()

    _worker_pipeline_loop(weights, group, is_last=True)

    # send_token must have been called once; value is from untrained model (noise).
    assert len(group.sent_tokens) == 1
    token_id, stop_flag = group.sent_tokens[0]
    assert isinstance(token_id, int)
    assert stop_flag is False

    # Must NOT have called send_hidden (it's the last rank).
    assert len(group.sent_hiddens) == 0

    # STOP must have been propagated.
    assert group.sent_stops == 1


def test_pipeline_rank0_prefill_unpadded(tmp_path: Path) -> None:
    """Rank-0 prefill send must have shape [1, prompt_tokens, hidden_dim], not
    [1, 1, hidden_dim] (old bug) and not [1, bucket_size, hidden_dim] (padded).

    _StubTokenizer.encode always returns [1, 2, 3], so prompt_tokens == 3.
    The nearest prefill bucket for 3 tokens is 32, so without the fix
    hidden.shape[1] would be 1 (old slice) or 32 (raw padded).
    After the fix it must be exactly 3.
    """
    from exo.worker.engines.tinygrad.generator.generate import (
        _rank0_pipeline_generate,  # pyright: ignore[reportPrivateUsage]
    )
    from exo.worker.engines.tinygrad.weight_loader import load_transformer_weights

    _build_fake_model(tmp_path)
    config = _make_config()
    weights = load_transformer_weights(tmp_path, config, is_first_rank=True, is_last_rank=False)

    tokenizer = _StubTokenizer()
    task = _make_task()

    group = _fake_group(rank=0, world_size=2)
    # One token then a stop-flagged token so the generator terminates quickly.
    group.queue_token(10, stop=False)
    group.queue_token(11, stop=True)

    list(_rank0_pipeline_generate(weights, tokenizer, task, "hello world", group))

    # The first send_hidden must be the prefill.
    assert len(group.sent_hiddens) >= 1, "expected at least one send_hidden call"
    prefill_hidden = group.sent_hiddens[0]
    prefill_shape: tuple[int, ...] = prefill_hidden.shape  # pyright: ignore[reportAny]

    # prompt_tokens == 3 (StubTokenizer always encodes to [1, 2, 3]).
    expected_prompt_tokens = 3
    assert prefill_shape[1] == expected_prompt_tokens, (
        f"prefill hidden shape[1] should be {expected_prompt_tokens} (prompt_tokens), "
        f"got {prefill_shape[1]} — was it still [1,1,H] (old slice) or [1,32,H] (padded)?"
    )
    assert prefill_shape[0] == 1, f"expected batch=1, got {prefill_shape[0]}"
    assert prefill_shape[2] == _HIDDEN, f"expected hidden_dim={_HIDDEN}, got {prefill_shape[2]}"
    assert prefill_hidden.dtype == np.float32, f"expected fp32, got {prefill_hidden.dtype}"


def test_worker_rank_generate_yields_nothing(tmp_path: Path) -> None:
    """tinygrad_generate on a non-rank-0 worker must yield nothing.

    The pipeline loop is patched out to avoid needing real sockets.
    """
    from exo.shared.types.worker.runner_response import GenerationResponse
    from exo.worker.engines.tinygrad.generator.generate import tinygrad_generate
    from exo.worker.engines.tinygrad.weight_loader import load_transformer_weights

    _build_fake_model(tmp_path)
    config = _make_config()
    weights = load_transformer_weights(tmp_path, config, is_first_rank=False, is_last_rank=True)

    tokenizer = _StubTokenizer()
    task = _make_task()

    group = _fake_group(rank=1, world_size=2)

    # Patch _worker_pipeline_loop so we don't need actual socket traffic.
    with patch(
        "exo.worker.engines.tinygrad.generator.generate._worker_pipeline_loop"
    ) as mock_loop:
        responses: list[GenerationResponse] = list(
            tinygrad_generate(weights, tokenizer, task, "hello", group=group)
        )

    assert responses == [], f"worker rank should yield nothing, got {responses}"
    mock_loop.assert_called_once_with(weights, group, is_last=True)

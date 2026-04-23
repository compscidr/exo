import contextlib
import struct
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from exo.worker.engines.tinygrad.pipeline_group import PipelineGroup

from tinygrad.dtype import dtypes
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import Context
from tinygrad.tensor import Tensor

from exo.shared.model_config import ModelConfig
from exo.shared.models.model_cards import ModelId
from exo.shared.tokenizer.eos_tokens import get_eos_token_ids_for_model
from exo.shared.types.api import (
    CompletionTokensDetails,
    GenerationStats,
    PromptTokensDetails,
    TopLogprobItem,
    Usage,
)
from exo.shared.types.memory import Memory
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.tinygrad.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)

from ..cache import KVCache
from ..forward import forward_pass
from ..sampling import sample_token
from ..weight_loader import TransformerWeights

_PREFILL_BUCKETS: list[int] = [32, 64, 128, 256, 512]

def _pad_to_bucket(input_ids: list[int], pad_id: int = 0) -> list[int]:
    """
        Padding input_ids to the nearest bucket size will cache the tensor
        sizes. This will increase the guarentee to hit the cache, leading to
        quicker time to first token.
    """

    for bucket in _PREFILL_BUCKETS:
        if len(input_ids) <= bucket:
            return input_ids + [pad_id] * (bucket - len(input_ids))
    return input_ids


@dataclass
class _JitState:
    """Persistent decode state reused across requests."""
    jit_decode: Callable[..., tuple[Tensor, ...]]
    cache: KVCache
    input_buffer: Tensor
    position_buffer: Tensor

_jit_registry: dict[int, _JitState] = {}

def cleanup_jit_state() -> None:
    """Called by engine cleanup to free all JIT state."""
    _jit_registry.clear()


def _build_jit_decode(
    weights: TransformerWeights,
    cache: KVCache,
) -> Callable[..., tuple[Tensor, ...]]:
    num_layers = len(weights.layers)

    @TinyJit
    def decode(
        input_ids: Tensor, position: Tensor,
        rope_cos_table: Tensor, rope_sin_table: Tensor,
        *cache_kv: Tensor,
    ) -> tuple[Tensor, ...]:
        for i in range(num_layers):
            cache.keys[i] = cache_kv[i]
            cache.values[i] = cache_kv[num_layers + i]

        logits, _ = forward_pass(
            weights, input_ids, cache,
            position_offset=position,
            rope_cos=rope_cos_table, rope_sin=rope_sin_table,
        )

        # Realize everything at once — JIT captures one fused kernel schedule
        # instead of 32 fragmented ones.
        logits = logits.realize(*cache.keys, *cache.values)

        return (logits, *cache.keys, *cache.values)

    return decode


def _build_worker_jit_decode(
    weights: TransformerWeights,
    cache: KVCache,
) -> Callable[..., tuple[Tensor, ...]]:
    """Build a JIT-captured decode step for a worker rank.

    Differs from ``_build_jit_decode`` in one respect: the ``hidden_wire``
    input is a uint16 buffer carrying bf16 bit patterns (the on-wire
    format). The first op in the JIT graph bitcasts back to bf16 and
    casts to the model's activation dtype so forward_pass sees the right
    tensor dtype. Everything downstream is identical to the rank-0/single
    JIT — forward_pass dispatches on whether ``weights.lm_head`` is
    populated to return either a hidden state (middle rank) or logits
    (last rank).
    """
    num_layers = len(weights.layers)
    assert weights.rope_cos is not None
    activation_dtype = weights.rope_cos.dtype

    @TinyJit
    def decode(
        hidden_wire: Tensor, position: Tensor,
        rope_cos_table: Tensor, rope_sin_table: Tensor,
        *cache_kv: Tensor,
    ) -> tuple[Tensor, ...]:
        x = hidden_wire.bitcast(dtypes.bfloat16).cast(activation_dtype)
        for i in range(num_layers):
            cache.keys[i] = cache_kv[i]
            cache.values[i] = cache_kv[num_layers + i]

        output, _ = forward_pass(
            weights, x, cache,
            position_offset=position,
            rope_cos=rope_cos_table, rope_sin=rope_sin_table,
        )
        output = output.realize(*cache.keys, *cache.values)

        return (output, *cache.keys, *cache.values)

    return decode


def _tensor_to_np(t: Tensor) -> "np.ndarray[Any, np.dtype[np.uint16]]":
    """Convert a tinygrad Tensor to a writable uint16-packed bf16 ndarray.

    Sequence on the device: ``.cast(bfloat16)`` does a proper dtype
    conversion, then ``.bitcast(uint16)`` reinterprets the bit pattern
    without a value conversion so we can hand numpy — which has no
    native bf16 — a dtype it understands. The receiving rank does the
    reverse bitcast back to bf16 and then casts to the activation dtype.
    """
    raw: Any = t.cast(dtypes.bfloat16).bitcast(dtypes.uint16).numpy()
    result: np.ndarray[Any, np.dtype[np.uint16]] = raw  # pyright: ignore[reportAny]
    return result


def _make_kv_cache(model: TransformerWeights) -> KVCache:
    """Allocate and realize a fresh KV cache for this rank's layers."""
    config = model.config
    num_layers = len(model.layers)
    cache = KVCache(
        num_layers=num_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_seq_len=min(config.max_position_embeddings, 4096),
    )
    for i in range(num_layers):
        cache.keys[i] = cache.keys[i].contiguous().realize()  # pyright: ignore[reportUnknownMemberType]
        cache.values[i] = cache.values[i].contiguous().realize()  # pyright: ignore[reportUnknownMemberType]
    return cache


def _single_rank_generate(
    model: TransformerWeights,
    tokenizer: Any,  # pyright: ignore[reportAny]
    task: TextGenerationTaskParams,
    prompt: str,
) -> Generator[GenerationResponse]:
    """Single-rank (no pipeline) generation — the original code path."""
    input_ids = _encode_prompt(tokenizer, prompt)

    max_tokens = task.max_output_tokens or DEFAULT_MAX_TOKENS
    temperature = task.temperature or DEFAULT_TEMPERATURE
    top_p = task.top_p or DEFAULT_TOP_P

    request_logprobs = task.logprobs
    top_logprobs_count = task.top_logprobs or 0

    eos_ids = _get_eos_ids(tokenizer, model.config)
    print(f"[DEBUG] eos_ids={eos_ids}")
    prompt_tokens = len(input_ids)
    input_ids = _pad_to_bucket(input_ids)

    model_key = id(model)
    num_layers = len(model.layers)
    state = _jit_registry.get(model_key)

    if state is None:
        # First request: create cache, JIT, and pre-allocate buffers
        cache = _make_kv_cache(model)
        jit_decode = _build_jit_decode(model, cache)
        input_buffer = Tensor.empty(1, 1, dtype=dtypes.int32).contiguous().realize()  # pyright: ignore[reportUnknownMemberType]
        position_buffer = Tensor.empty(1, dtype=dtypes.int32).contiguous().realize()  # pyright: ignore[reportUnknownMemberType]
        state = _JitState(
            jit_decode=jit_decode,
            cache=cache,
            input_buffer=input_buffer,
            position_buffer=position_buffer,
        )
        _jit_registry[model_key] = state

    cache = state.cache

    # Batched prefill: process all prompt tokens in a single forward pass.
    # Uses position_offset=0 (int) which triggers local attention (seq_len × seq_len)
    # instead of full-cache attention, producing ~324 kernel dispatches total
    # instead of 324 × N token-by-token dispatches.
    # BEAM is disabled because prefill shapes vary per prompt length (not cacheable)
    # and BEAM may select WMMA kernels incompatible with RDNA 2 (gfx1032).
    if not input_ids:
        raise ValueError("Prompt must contain at least one token")

    prefill_start = time.time()
    prompt_tensor = Tensor(input_ids, dtype=dtypes.int32).reshape(1, -1).contiguous().realize()  # pyright: ignore[reportUnknownMemberType]
    with Context(BEAM=0):
        logits, _ = forward_pass(
            model, prompt_tensor, cache,
            position_offset=0,
            rope_cos=model.rope_cos, rope_sin=model.rope_sin,
        )
        # Take the real last token's logits (not the padded last position).
        logits = logits[:, prompt_tokens - 1:prompt_tokens, :].contiguous()  # pyright: ignore[reportUnknownMemberType]
        # Make cache tensors contiguous for JIT compatibility.
        for i in range(num_layers):
            cache.keys[i] = cache.keys[i].contiguous()  # pyright: ignore[reportUnknownMemberType]
            cache.values[i] = cache.values[i].contiguous()  # pyright: ignore[reportUnknownMemberType]
        # Realize everything at once — same pattern as _build_jit_decode.
        logits = logits.realize(*cache.keys, *cache.values)

    # Rebuild the JIT after prefill. The batched prefill creates entirely new
    # cache tensor objects (via Tensor.where + contiguous + realize) that differ
    # from the JIT's captured output buffers. Rebuilding ensures the JIT
    # re-captures with the correct buffer objects. The cost is 2 slow decode
    # steps per request (cnt=0 jit-ignore, cnt=1 jit-capture), after which
    # all subsequent tokens use fast JIT replay.
    jit_decode = _build_jit_decode(model, cache)
    state.jit_decode = jit_decode

    prefill_time = time.time() - prefill_start
    prompt_tps = prompt_tokens / max(prefill_time, 1e-9)

    position = prompt_tokens

    # Decode
    generation_start = time.time()
    for token_idx in range(max_tokens):
        result = sample_token(
            logits, temperature=temperature, top_p=top_p,
            top_logprobs_count=top_logprobs_count,
            request_logprobs=request_logprobs,
        )

        token_text: str = tokenizer.decode([result.token_id])  # pyright: ignore[reportAny]

        is_eos = result.token_id in eos_ids
        if is_eos:
            print(f"[DEBUG] EOS detected: token_id={result.token_id}, text={token_text!r}")
        if "<|eot_id|>" in token_text or "<|end" in token_text:
            print(f"[DEBUG] Special token in text but is_eos={is_eos}, token_id={result.token_id}")
        tokens_generated = token_idx + 1
        elapsed = time.time() - generation_start
        generation_tps = tokens_generated / max(elapsed, 1e-9)

        finish_reason = None
        stats = None
        usage = None

        if is_eos:
            finish_reason = "stop"
        elif token_idx == max_tokens - 1:
            finish_reason = "length"

        if finish_reason is not None:
            stats = GenerationStats(
                prompt_tps=prompt_tps, generation_tps=generation_tps,
                prompt_tokens=prompt_tokens,
                generation_tokens=tokens_generated,
                peak_memory_usage=Memory.from_bytes(0),
            )

            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=tokens_generated,
                total_tokens=prompt_tokens + tokens_generated,
                prompt_tokens_details=PromptTokensDetails(),
                completion_tokens_details=CompletionTokensDetails(),
            )

        logprob = result.logprob if request_logprobs else None
        top_lps = None
        if request_logprobs and task.top_logprobs:
            top_lps = [
                TopLogprobItem(
                    token=str(tokenizer.decode([tok_id])),  # pyright: ignore[reportAny]
                    logprob=lp,
                    bytes=list(str(tokenizer.decode([tok_id])).encode("utf-8")),  # pyright: ignore[reportAny]
                )
                for tok_id, lp in result.top_logprobs
            ]

        if is_eos:
            token_text = ""

        yield GenerationResponse(
            text=token_text, token=result.token_id,
            logprob=logprob, top_logprobs=top_lps,
            finish_reason=finish_reason, stats=stats, usage=usage,
        )

        if finish_reason is not None:
            break

        state.input_buffer._buffer().copyin(memoryview(bytearray(struct.pack('=i', result.token_id))))  # pyright: ignore[reportPrivateUsage]
        state.position_buffer._buffer().copyin(memoryview(bytearray(struct.pack('=i', position))))  # pyright: ignore[reportPrivateUsage]
        results = jit_decode(
            state.input_buffer, state.position_buffer,
            model.rope_cos, model.rope_sin,
            *cache.keys, *cache.values,
        )

        logits = results[0]
        for i in range(num_layers):
            cache.keys[i] = results[1 + i]
            cache.values[i] = results[1 + num_layers + i]
        position += 1


def _worker_pipeline_loop(
    model: TransformerWeights,
    group: "PipelineGroup",
    is_last: bool,
) -> None:
    """Non-rank-0 pipeline worker loop.

    Receives hidden states from the previous rank, runs forward_pass on the
    local layer slice, then either:
    - sends the output hidden state to the next rank (middle ranks), or
    - samples a token and sends it back to rank 0 (last rank).

    Runs until a TAG_STOP is received, then propagates the stop signal and returns.
    """
    from exo.worker.engines.tinygrad.pipeline_group import (
        TAG_HIDDEN,
        TAG_STOP,
        decode_hidden,
    )

    cache = _make_kv_cache(model)

    position: int = 0
    first = True
    num_layers = len(model.layers)
    hidden_dim = model.config.hidden_size

    # JIT state — built lazily after the prefill (first HIDDEN arrival).
    # Prefill stays non-JIT because its shape varies per prompt length.
    jit_decode: Callable[..., tuple[Tensor, ...]] | None = None
    hidden_buf: Tensor | None = None
    position_buf: Tensor | None = None

    while True:
        tag, payload = group.recv_any()

        if tag == TAG_STOP:
            # Propagate stop downstream and exit.
            group.send_stop()
            return

        if tag == TAG_HIDDEN:
            _position_offset, arr = decode_hidden(payload)
            # arr shape: [batch, seq_len, hidden_dim]
            arr_shape: tuple[int, ...] = arr.shape  # pyright: ignore[reportAny]
            seq_len = int(arr_shape[1])

            if first:
                # Prefill path. Shape varies per prompt so no JIT yet;
                # BEAM is disabled because each prompt bucket hits a new
                # kernel shape.
                hidden = (
                    Tensor(arr)
                    .bitcast(dtypes.bfloat16)
                    .cast(model.rope_cos.dtype)  # pyright: ignore[reportUnknownMemberType]
                    .contiguous()
                    .realize()
                )
                with Context(BEAM=0):
                    output, _ = forward_pass(
                        model, hidden, cache,
                        position_offset=0,
                        rope_cos=model.rope_cos, rope_sin=model.rope_sin,
                    )
                    for i in range(len(cache.keys)):
                        cache.keys[i] = cache.keys[i].contiguous()  # pyright: ignore[reportUnknownMemberType]
                        cache.values[i] = cache.values[i].contiguous()  # pyright: ignore[reportUnknownMemberType]
                    output = output.contiguous().realize(*cache.keys, *cache.values)  # pyright: ignore[reportUnknownMemberType]

                # Build JIT and persistent buffers for the decode phase.
                # Must happen *after* prefill's contiguous+realize dance —
                # prefill creates new cache tensor objects and the JIT
                # captures the post-prefill ones as its baseline.
                jit_decode = _build_worker_jit_decode(model, cache)
                hidden_buf = Tensor.empty(1, 1, hidden_dim, dtype=dtypes.uint16).contiguous().realize()  # pyright: ignore[reportUnknownMemberType]
                position_buf = Tensor.empty(1, dtype=dtypes.int32).contiguous().realize()  # pyright: ignore[reportUnknownMemberType]
            else:
                # Decode path (JIT replay). seq_len is always 1 here; the
                # JIT captures that shape. Copy the raw uint16 wire bytes
                # into hidden_buf's storage — bitcast+cast to activation
                # dtype happens inside the JIT graph.
                assert jit_decode is not None
                assert hidden_buf is not None
                assert position_buf is not None
                hidden_buf._buffer().copyin(memoryview(bytearray(arr.tobytes())))  # pyright: ignore[reportPrivateUsage]
                position_buf._buffer().copyin(memoryview(bytearray(struct.pack("=i", position))))  # pyright: ignore[reportPrivateUsage]
                results = jit_decode(
                    hidden_buf, position_buf,
                    model.rope_cos, model.rope_sin,
                    *cache.keys, *cache.values,
                )
                output = results[0]
                for i in range(num_layers):
                    cache.keys[i] = results[1 + i]
                    cache.values[i] = results[1 + num_layers + i]

            if is_last:
                # Last rank: sample a token from the logits and send back to rank 0.
                token_result = sample_token(output, temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P)
                group.send_token(token_result.token_id, stop=False)
            else:
                # Middle rank: convert hidden state to fp16 numpy and forward downstream.
                hidden_np = _tensor_to_np(output)
                group.send_hidden(hidden_np)

            position += seq_len
            first = False

        else:
            # TAG_TOKEN is unexpected on a worker rank in this design.
            import warnings
            warnings.warn(
                f"[pipeline worker rank {group.rank}] received unexpected tag={tag}; ignoring",
                stacklevel=1,
            )


def _rank0_pipeline_generate(
    model: TransformerWeights,
    tokenizer: Any,  # pyright: ignore[reportAny]
    task: TextGenerationTaskParams,
    prompt: str,
    group: "PipelineGroup",
) -> Generator[GenerationResponse]:
    """Rank-0 pipeline generator.

    Handles prefill locally, ships the hidden state downstream, waits for the
    last rank to sample a token, then drives the decode loop.  Wraps everything
    in try/finally so that the pipeline workers always receive a STOP signal even
    if the caller drops the generator early (e.g., warmup after N tokens).
    """
    input_ids = _encode_prompt(tokenizer, prompt)

    max_tokens = task.max_output_tokens or DEFAULT_MAX_TOKENS
    # temperature, top_p, and logprob settings are not used on rank 0:
    # rank 0 has no lm_head so it cannot sample; sampling is done by the last rank.

    eos_ids = _get_eos_ids(tokenizer, model.config)
    prompt_tokens = len(input_ids)
    input_ids = _pad_to_bucket(input_ids)

    if not input_ids:
        raise ValueError("Prompt must contain at least one token")

    cache = _make_kv_cache(model)

    prefill_start = time.time()

    try:
        # ── Prefill ──────────────────────────────────────────────────────────
        prompt_tensor = Tensor(input_ids, dtype=dtypes.int32).reshape(1, -1).contiguous().realize()  # pyright: ignore[reportUnknownMemberType]
        with Context(BEAM=0):
            hidden_or_logits, _ = forward_pass(
                model, prompt_tensor, cache,
                position_offset=0,
                rope_cos=model.rope_cos, rope_sin=model.rope_sin,
            )
            # Rank 0 is never the last rank in multi-rank mode (lm_head is None),
            # so this is always a hidden state.
            # Materialize the output and the whole cache in one scheduling
            # pass — matches the single-rank prefill pattern and prevents
            # stale/lazy cache tensors from corrupting later decode steps.
            for i in range(len(cache.keys)):
                cache.keys[i] = cache.keys[i].contiguous()  # pyright: ignore[reportUnknownMemberType]
                cache.values[i] = cache.values[i].contiguous()  # pyright: ignore[reportUnknownMemberType]
            hidden_or_logits = hidden_or_logits.contiguous().realize(*cache.keys, *cache.values)  # pyright: ignore[reportUnknownMemberType]

        # Ship only the real (un-padded) prefill positions to downstream ranks.
        # Slicing to [:, :prompt_tokens, :] strips the bucket-padding tokens so
        # that each downstream rank's forward_pass populates exactly prompt_tokens
        # KV-cache entries (positions 0..prompt_tokens-1), matching rank 0's cache.
        # Sending only the last token would leave downstream KV caches with a
        # single entry and produce garbage attention during decode.
        prefill_hidden = hidden_or_logits[:, :prompt_tokens, :]
        prefill_hidden = prefill_hidden.contiguous().realize()  # pyright: ignore[reportUnknownMemberType]

        prefill_np = _tensor_to_np(prefill_hidden)
        group.send_hidden(prefill_np)

        # Wait for last rank to sample the first token.
        token_id, stop = group.recv_token()

        prefill_time = time.time() - prefill_start
        prompt_tps = prompt_tokens / max(prefill_time, 1e-9)

        position = prompt_tokens

        # ── JIT setup for the decode loop ────────────────────────────────────
        # Build the JIT against the cache *after* the prefill's contiguous+
        # realize dance: prefill creates new cache tensor objects, so a JIT
        # built before prefill would capture the wrong buffers. Persistent
        # input buffers get `_buffer().copyin()`'d each decode step so the
        # JIT can replay without reallocating argument tensors.
        num_layers = len(model.layers)
        jit_decode = _build_jit_decode(model, cache)
        input_buf = Tensor.empty(1, 1, dtype=dtypes.int32).contiguous().realize()  # pyright: ignore[reportUnknownMemberType]
        position_buf = Tensor.empty(1, dtype=dtypes.int32).contiguous().realize()  # pyright: ignore[reportUnknownMemberType]

        # ── Decode loop ───────────────────────────────────────────────────────
        generation_start = time.time()
        # Accumulators to break down per-decode wall clock. Logged every N
        # tokens so we can see whether rank-0 local forward, the outbound
        # send, or the worker-roundtrip dominates.
        _inst_local_forward_s = 0.0
        _inst_send_s = 0.0
        _inst_recv_s = 0.0
        _inst_samples = 0
        _inst_every = 20
        for token_idx in range(max_tokens):
            token_text: str = tokenizer.decode([token_id])  # pyright: ignore[reportAny]

            is_eos = token_id in eos_ids
            tokens_generated = token_idx + 1
            elapsed = time.time() - generation_start
            generation_tps = tokens_generated / max(elapsed, 1e-9)

            finish_reason = None
            stats = None
            usage = None

            if is_eos or stop:
                finish_reason = "stop"
            elif token_idx == max_tokens - 1:
                finish_reason = "length"

            if finish_reason is not None:
                stats = GenerationStats(
                    prompt_tps=prompt_tps, generation_tps=generation_tps,
                    prompt_tokens=prompt_tokens,
                    generation_tokens=tokens_generated,
                    peak_memory_usage=Memory.from_bytes(0),
                )
                usage = Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=tokens_generated,
                    total_tokens=prompt_tokens + tokens_generated,
                    prompt_tokens_details=PromptTokensDetails(),
                    completion_tokens_details=CompletionTokensDetails(),
                )

            if is_eos:
                token_text = ""

            # Logprobs are not available on rank 0 (no lm_head).
            yield GenerationResponse(
                text=token_text, token=token_id,
                logprob=None, top_logprobs=None,
                finish_reason=finish_reason, stats=stats, usage=usage,
            )

            if finish_reason is not None:
                # Return; the enclosing try/finally will send exactly one STOP
                # and drain the ring-echo. Sending STOP here too would leave
                # a second STOP in the downstream rank's recv buffer that the
                # *next* request would consume before its own prefill-hidden,
                # causing the next request's worker loop to exit in ~ms.
                return

            # ── Decode step (JIT): embed single token, ship hidden ────
            _inst_t0 = time.perf_counter()
            input_buf._buffer().copyin(memoryview(bytearray(struct.pack("=i", token_id))))  # pyright: ignore[reportPrivateUsage]
            position_buf._buffer().copyin(memoryview(bytearray(struct.pack("=i", position))))  # pyright: ignore[reportPrivateUsage]
            results = jit_decode(
                input_buf, position_buf,
                model.rope_cos, model.rope_sin,
                *cache.keys, *cache.values,
            )
            decode_hidden = results[0]
            for i in range(num_layers):
                cache.keys[i] = results[1 + i]
                cache.values[i] = results[1 + num_layers + i]
            _inst_t1 = time.perf_counter()

            decode_np = _tensor_to_np(decode_hidden)
            group.send_hidden(decode_np)
            _inst_t2 = time.perf_counter()

            token_id, stop = group.recv_token()
            _inst_t3 = time.perf_counter()
            position += 1

            _inst_local_forward_s += _inst_t1 - _inst_t0
            _inst_send_s += _inst_t2 - _inst_t1
            _inst_recv_s += _inst_t3 - _inst_t2
            _inst_samples += 1
            if _inst_samples >= _inst_every:
                import sys as _sys
                total_s = _inst_local_forward_s + _inst_send_s + _inst_recv_s
                print(
                    f"[pipeline rank 0 decode {_inst_every}-token avg] "
                    f"local_fwd={1000*_inst_local_forward_s/_inst_every:.1f}ms "
                    f"send={1000*_inst_send_s/_inst_every:.1f}ms "
                    f"worker_rtt={1000*_inst_recv_s/_inst_every:.1f}ms "
                    f"total={1000*total_s/_inst_every:.1f}ms "
                    f"tps={_inst_every/max(total_s, 1e-9):.2f}",
                    file=_sys.stderr, flush=True,
                )
                _inst_local_forward_s = 0.0
                _inst_send_s = 0.0
                _inst_recv_s = 0.0
                _inst_samples = 0

    finally:
        from exo.worker.engines.tinygrad.pipeline_group import TAG_STOP

        # Ensure workers are unblocked even if the generator is closed early.
        with contextlib.suppress(Exception):
            group.send_stop()
        # The STOP we just sent propagates through the ring and lands back in
        # this rank's recv_sock (each worker forwards STOP before exiting, so
        # in any ring size rank 0 receives exactly one STOP echo).  Drain it
        # now so the next session starts with an empty recv buffer.
        with contextlib.suppress(Exception):
            while True:
                tag, _payload = group.recv_any()
                if tag == TAG_STOP:
                    break


def tinygrad_generate(
    model: TransformerWeights,
    tokenizer: Any,  # pyright: ignore[reportAny]
    task: TextGenerationTaskParams,
    prompt: str,
    kv_prefix_cache: Any = None,  # pyright: ignore[reportAny]
    on_prefill_progress: Callable[[int, int], None] | None = None,
    group: "PipelineGroup | None" = None,
) -> Generator[GenerationResponse]:
    if group is None or group.world_size == 1:
        yield from _single_rank_generate(model, tokenizer, task, prompt)
        return

    if group.rank == 0:
        yield from _rank0_pipeline_generate(model, tokenizer, task, prompt, group)
        return

    # Non-rank-0 worker: run the blocking pipeline loop (no yields).
    _worker_pipeline_loop(model, group, is_last=(group.rank == group.world_size - 1))
    # Return without yielding — the caller's `for response in gen:` is a no-op.


def warmup_inference(model: TransformerWeights, tokenizer: Any, group: "PipelineGroup | None" = None) -> int:  # pyright: ignore[reportAny]
    """Run a full generation loop to warm up forward pass, KV cache, and sampling."""
    from exo.shared.tokenizer.chat_template import apply_chat_template
    from exo.shared.types.common import ModelId as CommonModelId
    from exo.shared.types.text_generation import InputMessage

    warmup_task = TextGenerationTaskParams(
        model=CommonModelId("warmup"),
        input=[InputMessage(role="user", content="Time to warm up!")],
    )

    prompt: str = apply_chat_template(tokenizer, warmup_task)
    tokens_generated = 0

    for _ in tinygrad_generate(model, tokenizer, warmup_task, prompt, group=group):
        tokens_generated += 1
        if tokens_generated >= 5:
            break

    # Only rank 0 owns embed_tokens, so only rank 0 benefits from prefill bucket warmup.
    if group is None or group.rank == 0:
        _warmup_prefill_buckets(model)

    return tokens_generated


def _warmup_prefill_buckets(model: TransformerWeights) -> None:
    """Pre-compile prefill kernels at each bucket size to avoid first-request compilation."""
    model_key = id(model)
    state = _jit_registry.get(model_key)
    if state is None:
        return

    cache = state.cache
    num_layers = len(model.layers)

    with Context(BEAM=0):
        for bucket_size in _PREFILL_BUCKETS:
            dummy = Tensor.zeros(1, bucket_size, dtype=dtypes.int32).contiguous().realize()  # pyright: ignore[reportUnknownMemberType]
            logits, _ = forward_pass(
                model, dummy, cache,
                position_offset=0,
                rope_cos=model.rope_cos, rope_sin=model.rope_sin,
            )
            logits = logits[:, -1:, :].contiguous()  # pyright: ignore[reportUnknownMemberType]
            for i in range(num_layers):
                cache.keys[i] = cache.keys[i].contiguous()  # pyright: ignore[reportUnknownMemberType]
                cache.values[i] = cache.values[i].contiguous()  # pyright: ignore[reportUnknownMemberType]
            logits.realize(*cache.keys, *cache.values)

def _encode_prompt(tokenizer: Any, prompt: str) -> list[int]:  # pyright: ignore[reportAny]
    result: Any = tokenizer.encode(prompt)  # pyright: ignore[reportAny]

    return result.ids if hasattr(result, "ids") else result  # pyright: ignore[reportAny]

def _get_eos_ids(tokenizer: Any, config: ModelConfig) -> set[int]:  # pyright: ignore[reportAny]
    eos_ids: set[int] = set()

    model_eos = get_eos_token_ids_for_model(ModelId(config.architecture_spec.name))

    if model_eos:
        eos_ids.update(model_eos)

    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:  # pyright: ignore[reportAny]
        eos_ids.add(int(tokenizer.eos_token_id))  # pyright: ignore[reportAny]

    if not eos_ids:
        eos_ids.add(2)

    return eos_ids

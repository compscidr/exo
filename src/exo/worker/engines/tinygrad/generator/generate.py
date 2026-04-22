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

    while True:
        tag, payload = group.recv_any()

        if tag == TAG_STOP:
            # Propagate stop downstream and exit.
            group.send_stop()
            return

        if tag == TAG_HIDDEN:
            arr = decode_hidden(payload)
            # arr shape: [batch, seq_len, hidden_dim]
            arr_shape: tuple[int, ...] = arr.shape  # pyright: ignore[reportAny]
            seq_len = int(arr_shape[1])

            # The wire format is bf16 packed as uint16 (numpy has no bf16).
            # Reinterpret the bit pattern as bf16, then cast to the model's
            # internal activation dtype (matches rope_cos) so forward_pass's
            # matmuls don't hit a dtype mismatch.
            hidden = (
                Tensor(arr)
                .bitcast(dtypes.bfloat16)
                .cast(model.rope_cos.dtype)  # pyright: ignore[reportUnknownMemberType]
                .contiguous()
                .realize()
            )

            # Prefill: pass position_offset=0 (int) so attention uses the
            # local-only seq_len×seq_len path (correct and efficient here).
            # Decode (first=False, seq_len=1): pass position_offset as a
            # Tensor so attention takes the cache-reading branch — otherwise
            # it ignores all prefill/prior-decode entries and produces garbage.
            position_offset: "int | Tensor"
            if first:
                position_offset = 0
            else:
                position_offset = Tensor([position], dtype=dtypes.int32).contiguous().realize()  # pyright: ignore[reportUnknownMemberType]

            # Prefill shapes vary per prompt length and aren't cacheable,
            # so disable BEAM. Decode shape is fixed (seq_len=1) so let
            # BEAM search kernels like the runner bootstrap defaults to.
            fwd_context: "contextlib.AbstractContextManager[object]" = (
                Context(BEAM=0) if first else contextlib.nullcontext()
            )
            with fwd_context:
                output, _ = forward_pass(
                    model, hidden, cache,
                    position_offset=position_offset,
                    rope_cos=model.rope_cos, rope_sin=model.rope_sin,
                )
                for i in range(len(cache.keys)):
                    cache.keys[i] = cache.keys[i].contiguous()  # pyright: ignore[reportUnknownMemberType]
                    cache.values[i] = cache.values[i].contiguous()  # pyright: ignore[reportUnknownMemberType]
                output = output.contiguous().realize(*cache.keys, *cache.values)  # pyright: ignore[reportUnknownMemberType]

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

        # ── Decode loop ───────────────────────────────────────────────────────
        generation_start = time.time()
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
                # Send stop to workers and end.
                group.send_stop()
                return

            # ── Decode step: embed single token on rank 0, ship hidden ────
            tok_tensor = Tensor([[token_id]], dtype=dtypes.int32).contiguous().realize()  # pyright: ignore[reportUnknownMemberType]
            # CRITICAL: pass position_offset as a Tensor (not int) so
            # grouped_query_attention takes the "decode" branch that attends
            # against cache.keys/values — otherwise it attends only to the
            # current single token's K/V and ignores all prefill context.
            position_tensor = Tensor([position], dtype=dtypes.int32).contiguous().realize()  # pyright: ignore[reportUnknownMemberType]
            # Decode shape is fixed (seq_len=1), so BEAM can cache kernels;
            # let the runner-bootstrap default (BEAM=2) apply instead of
            # disabling BEAM as we used to.
            decode_hidden, _ = forward_pass(
                model, tok_tensor, cache,
                position_offset=position_tensor,
                rope_cos=model.rope_cos, rope_sin=model.rope_sin,
            )
            for i in range(len(cache.keys)):
                cache.keys[i] = cache.keys[i].contiguous()  # pyright: ignore[reportUnknownMemberType]
                cache.values[i] = cache.values[i].contiguous()  # pyright: ignore[reportUnknownMemberType]
            decode_hidden = decode_hidden.contiguous().realize(*cache.keys, *cache.values)  # pyright: ignore[reportUnknownMemberType]

            decode_np = _tensor_to_np(decode_hidden)
            group.send_hidden(decode_np)

            token_id, stop = group.recv_token()
            position += 1

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

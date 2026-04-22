from collections.abc import Callable
from pathlib import Path
from typing import Literal, NamedTuple, overload

from tinygrad.device import Device
from tinygrad.nn.state import safe_load
from tinygrad.tensor import Tensor

from exo.shared.architecture import ArchitectureSpec
from exo.shared.model_config import ModelConfig
from exo.worker.engines.tinygrad.layers.rotary import compute_rope_frequencies
from exo.worker.engines.tinygrad.quantization.layers import (
    QuantizedEmbedding,
    QuantizedLinear,
)
from exo.worker.engines.tinygrad.quantization.packing import PackedTensor
from exo.worker.engines.tinygrad.quantization.shapes import infer_weight_shape

LinearWeight = Tensor | QuantizedLinear
EmbedWeight = Tensor | QuantizedEmbedding

class LayerWeights(NamedTuple):
    qkv_proj: LinearWeight       # Merged Q+K+V
    o_proj: LinearWeight

    gate_up_proj: LinearWeight   # Merged gate+up
    down_proj: LinearWeight

    input_norm: Tensor
    post_attn_norm: Tensor

    # Optional layers
    q_norm: Tensor | None = None
    k_norm: Tensor | None = None

    # MoE (None for dense models)
    router_weight: Tensor | None = None
    expert_gate_projs: list[LinearWeight] | None = None
    expert_up_projs: list[LinearWeight] | None = None
    expert_down_projs: list[LinearWeight] | None = None

class TransformerWeights(NamedTuple):
    embed_tokens: EmbedWeight | None
    lm_head: LinearWeight | None
    final_norm: Tensor | None
    layers: list[LayerWeights]
    config: ModelConfig
    rope_sin: Tensor
    rope_cos: Tensor

def load_transformer_weights(
    model_path: Path,
    config: ModelConfig,
    start_layer: int = 0,
    end_layer: int | None = None,
    is_first_rank: bool = True,
    is_last_rank: bool = True,
) -> TransformerWeights:

    if end_layer is None:
        end_layer = config.num_hidden_layers

    spec = config.architecture_spec

    # Build a key predicate so _load_all_safetensors only loads the weights
    # this rank actually uses — critical for pipeline-parallel setups where
    # a single GPU cannot hold the full model's safetensors on device.
    # NB: append a trailing "." to each layer prefix so "model.layers.1."
    # doesn't also match "model.layers.10.self_attn.q_proj.weight" etc.
    layer_prefixes: list[str] = [
        f"{spec.layer_prefix.format(layer_idx=i)}." for i in range(start_layer, end_layer)
    ]
    boundary_prefixes: list[str] = []
    needs_embed = is_first_rank or (is_last_rank and config.tie_word_embeddings)
    if needs_embed:
        boundary_prefixes.append(f"{spec.embed_key}.")
    if is_last_rank:
        boundary_prefixes.append(f"{spec.final_norm_key}.")
        if not config.tie_word_embeddings:
            boundary_prefixes.append(f"{spec.lm_head_key}.")

    def _keep_key(key: str) -> bool:
        for p in layer_prefixes:
            if key.startswith(p):
                return True
        for p in boundary_prefixes:
            if key.startswith(p):
                return True
        return False

    raw_weights = _load_all_safetensors(model_path, keep_predicate=_keep_key)

    embed_tokens: EmbedWeight | None = None
    lm_head: LinearWeight | None = None
    final_norm: Tensor | None = None

    # embed_tokens only needed on rank 0.
    if is_first_rank:
        embed_tokens = _build_weight(
            raw_weights, f"{spec.embed_key}.weight", config, is_embedding=True
        )

    # lm_head + final_norm only needed on the last rank.
    if is_last_rank:
        final_norm = raw_weights[f"{spec.final_norm_key}.weight"]

        if config.tie_word_embeddings:
            # lm_head shares weights with embed_tokens. If we already loaded
            # embed_tokens above, reuse it; otherwise load the embed weight here
            # purely to source lm_head.
            source_embed: EmbedWeight
            if embed_tokens is not None:
                source_embed = embed_tokens
            else:
                source_embed = _build_weight(
                    raw_weights, f"{spec.embed_key}.weight", config, is_embedding=True
                )
            if isinstance(source_embed, QuantizedEmbedding):
                lm_head = QuantizedLinear(
                    weight_q=source_embed.weight_q,
                    scales=source_embed.scales,
                    biases=source_embed.biases,
                    group_size=source_embed.group_size,
                )
            else:
                lm_head = source_embed
        else:
            lm_head = _build_weight(
                raw_weights, f"{spec.lm_head_key}.weight", config,
            )

    layers: list[LayerWeights] = []

    for layer_idx in range(start_layer, end_layer):
        prefix = spec.layer_prefix.format(layer_idx=layer_idx)
        layers.append(_build_layer_weights(raw_weights, prefix, spec, config))
        # Drop this layer's raw keys once its merged weights are built.
        # _build_layer_weights produces merged qkv_proj/gate_up_proj tensors
        # that hold new storage; the originals in raw_weights (per-layer
        # q/k/v/gate/up/down/o/norms) are no longer needed. Keeping them alive
        # doubles peak VRAM on big pipelines — on a 14B/8bit model a beast
        # rank loading ~10 GB of raw + ~5 GB of merged per-layer accumulators
        # blows past 16 GB during the build loop.
        stale_prefix = f"{prefix}."
        for k in [kk for kk in raw_weights if kk.startswith(stale_prefix)]:
            del raw_weights[k]

    rope_cos, rope_sin = compute_rope_frequencies(
        head_dim=config.head_dim,
        max_seq_len=config.max_position_embeddings,
        rope_theta=config.rope_theta,
    )

    return TransformerWeights(
        embed_tokens=embed_tokens,
        lm_head=lm_head,
        final_norm=final_norm,
        layers=layers,
        config=config,
        rope_cos=rope_cos.realize(),
        rope_sin=rope_sin.realize(),
    )

def _merge_linear_weights(*weights: LinearWeight) -> LinearWeight:
    """Merge multiple LinearWeight objects by concatenating along the output dimension (dim 0).

    For quantized weights, concatenates packed uint32, scales, and biases directly —
    no extra memory from dequantization. For non-quantized or mixed weights, dequantizes
    first then concatenates plain Tensors.
    """
    if all(isinstance(w, QuantizedLinear) for w in weights):
        qls = [w for w in weights if isinstance(w, QuantizedLinear)]
        merged_tensor = qls[0].weight_q.tensor.cat(
            *[w.weight_q.tensor for w in qls[1:]], dim=0
        ).contiguous().realize()  # pyright: ignore[reportUnknownMemberType]
        merged_scales = qls[0].scales.cat(
            *[w.scales for w in qls[1:]], dim=0
        ).contiguous().realize()  # pyright: ignore[reportUnknownMemberType]
        merged_biases = qls[0].biases.cat(
            *[w.biases for w in qls[1:]], dim=0
        ).contiguous().realize()  # pyright: ignore[reportUnknownMemberType]
        return QuantizedLinear(
            weight_q=PackedTensor(
                tensor=merged_tensor,
                original_shape=(
                    sum(w.weight_q.original_shape[0] for w in qls),
                    qls[0].weight_q.original_shape[1],
                ),
                pack_factor=qls[0].weight_q.pack_factor,
                bits=qls[0].weight_q.bits,
            ),
            scales=merged_scales,
            biases=merged_biases,
            group_size=qls[0].group_size,
        )
    # Non-quantized or mixed: dequantize if needed, then cat
    tensors: list[Tensor] = []
    for w in weights:
        if isinstance(w, QuantizedLinear):
            tensors.append(w.dequantize())
        else:
            tensors.append(w)
    return tensors[0].cat(*tensors[1:], dim=0).contiguous().realize()  # pyright: ignore[reportUnknownMemberType]

def _build_layer_weights(
    raw: dict[str, Tensor],
    prefix: str,
    spec: ArchitectureSpec,
    config: ModelConfig,
) -> LayerWeights:
    def key(suffix: str) -> str:
        return f"{prefix}.{suffix}.weight"

    q_norm = raw.get(f"{prefix}.{spec.q_norm_key}.weight") if spec.q_norm_key else None
    k_norm = raw.get(f"{prefix}.{spec.k_norm_key}.weight") if spec.k_norm_key else None

    q_proj = _build_weight(raw, key(spec.q_proj_key), config)
    k_proj = _build_weight(raw, key(spec.k_proj_key), config)
    v_proj = _build_weight(raw, key(spec.v_proj_key), config)
    qkv_proj = _merge_linear_weights(q_proj, k_proj, v_proj)

    gate_proj = _build_weight(raw, key(spec.gate_proj_key), config)
    up_proj = _build_weight(raw, key(spec.up_proj_key), config)
    gate_up_proj = _merge_linear_weights(gate_proj, up_proj)

    return LayerWeights(
        qkv_proj=qkv_proj,
        o_proj=_build_weight(raw, key(spec.o_proj_key), config),
        gate_up_proj=gate_up_proj,
        down_proj=_build_weight(raw, key(spec.down_proj_key), config),
        input_norm=raw[f"{prefix}.{spec.input_norm_key}.weight"],
        post_attn_norm=raw[f"{prefix}.{spec.post_attn_norm_key}.weight"],
        q_norm=q_norm,
        k_norm=k_norm,
    )

@overload
def _build_weight(raw: dict[str, Tensor], key: str, config: ModelConfig, is_embedding: Literal[True]) -> EmbedWeight: ...
@overload
def _build_weight(raw: dict[str, Tensor], key: str, config: ModelConfig, is_embedding: Literal[False] = ...) -> LinearWeight: ...

def _build_weight(
    raw: dict[str, Tensor],
    key: str,
    config: ModelConfig,
    is_embedding: bool = False,
) -> LinearWeight | EmbedWeight:
    scales_key = key.replace(".weight", ".scales")
    biases_key = key.replace(".weight", ".biases")

    # MLX quantized: .weight (packed uint32) + .scales + .biases + quantization_config
    if key in raw and config.quantization_config is not None and scales_key in raw and biases_key in raw:
        qcfg = config.quantization_config
        packed = PackedTensor(
            tensor = raw[key],
            original_shape = infer_weight_shape(key, config),
            pack_factor = 32 // qcfg.bits,
            bits = qcfg.bits,
        )

        if is_embedding:
            return QuantizedEmbedding(
                num_embeddings = config.vocab_size,
                embedding_dim = config.hidden_size,
                weight_q = packed,
                scales = raw[scales_key],
                biases = raw[biases_key],
                group_size = qcfg.group_size,
            )

        return QuantizedLinear(
            weight_q = packed,
            scales = raw[scales_key],
            biases = raw[biases_key],
            group_size = qcfg.group_size,
        )

    # Plain unquantized: .weight only
    if key in raw:
        return raw[key]

    # Legacy .qweight format
    qweight_key = key.replace(".weight", ".qweight")

    if qweight_key in raw and config.quantization_config is not None:
        qcfg = config.quantization_config
        packed = PackedTensor(
            tensor = raw[qweight_key],
            original_shape = infer_weight_shape(key, config),
            pack_factor = 32 // qcfg.bits,
            bits = qcfg.bits,
        )

        if is_embedding:
            return QuantizedEmbedding(
                num_embeddings = config.vocab_size,
                embedding_dim = config.hidden_size,
                weight_q = packed,
                scales = raw[scales_key],
                biases = raw[biases_key],
                group_size = qcfg.group_size,
            )

        return QuantizedLinear(
            weight_q = packed,
            scales = raw[scales_key],
            biases = raw[biases_key],
            group_size = qcfg.group_size,
        )

    raise KeyError(f"Weight key '{key}' not found (also tried {qweight_key})")

def _load_all_safetensors(
    path: Path,
    keep_predicate: "Callable[[str], bool] | None" = None,
) -> dict[str, Tensor]:
    from tinygrad.helpers import Context

    merged: dict[str, Tensor] = {}
    any_file = False

    # Disable BEAM during weight loading. Copy kernels (DISK -> GPU) have unique
    # shapes per tensor and don't benefit from beam search optimisation.
    # When `keep_predicate` is provided, only keys that pass the predicate are
    # realized on the default device — everything else is skipped entirely,
    # which is essential for pipeline-parallel setups where a single rank's
    # GPU can't hold the full model (e.g. 32B-4bit at ~16GB on a 10–16GB card).
    with Context(BEAM=0):
        for safetensor_file in sorted(path.glob("*.safetensors")):
            any_file = True
            shard = safe_load(str(safetensor_file))
            for key, tensor in shard.items():
                if keep_predicate is not None and not keep_predicate(key):
                    continue
                merged[key] = tensor.to(Device.DEFAULT).contiguous().realize()  # pyright: ignore[reportUnknownMemberType]

    if not any_file:
        raise FileNotFoundError(f"No .safetensors file found in {path}")

    return merged

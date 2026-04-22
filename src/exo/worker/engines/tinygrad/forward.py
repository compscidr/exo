from typing import NamedTuple

from tinygrad.tensor import Tensor

from exo.shared.model_config import ModelConfig
from exo.worker.engines.tinygrad.cache import KVCache
from exo.worker.engines.tinygrad.layers.attention import grouped_query_attention
from exo.worker.engines.tinygrad.layers.embedding import apply_embedding, apply_lm_head
from exo.worker.engines.tinygrad.layers.mlp import swiglu_mlp
from exo.worker.engines.tinygrad.layers.normalization import rms_norm
from exo.worker.engines.tinygrad.weight_loader import LayerWeights, TransformerWeights


class TransformerBlockBuilder(NamedTuple):
    cos_freqs: Tensor
    sin_freqs: Tensor
    layer: LayerWeights
    idx: int
    offset: int | Tensor

def forward_pass(
    weights: TransformerWeights,
    input_or_hidden: Tensor,
    cache: KVCache | None,
    position_offset: int | Tensor = 0,
    rope_cos: Tensor | None = None,
    rope_sin: Tensor | None = None,
) -> tuple[Tensor, KVCache]:
    """Run a forward pass through this rank's portion of the transformer.

    ``input_or_hidden`` is interpreted based on whether ``weights.embed_tokens``
    is present:

    * **Rank 0 / single-rank** (``embed_tokens is not None``): ``input_or_hidden``
      must be an int32 token-ID tensor of shape ``[batch, seq_len]``.  Embedding
      is applied and the resulting hidden state is fed to the layer loop.

    * **Middle / last pipeline rank** (``embed_tokens is None``): ``input_or_hidden``
      must already be an embedded hidden-state tensor of shape
      ``[batch, seq_len, hidden_dim]``.  The layer loop starts directly from it.

    The return value similarly depends on the rank:

    * **Last rank / single-rank** (``lm_head is not None``): returns logits of
      shape ``[batch, seq_len, vocab_size]``.
    * **Middle / first-only rank** (``lm_head is None``): returns the raw hidden
      state of shape ``[batch, seq_len, hidden_dim]`` for the pipeline transport
      to forward to the next rank.
    """
    config = weights.config

    if weights.embed_tokens is not None:
        # Rank 0 (or single-rank): run embedding on token IDs.
        x = apply_embedding(weights.embed_tokens, input_or_hidden)
    else:
        # Middle/last pipeline rank: input is already an embedded hidden state.
        x = input_or_hidden

    if cache is None:
        """
            I am reducing the max_seq_len down to 4096 to work
            with consumer grade GPUs. Unlike Apple systems,
            most computers have memory statically partionined
            if using integrated memory. With discrete GPUs, the
            VRAM issue becomes explicit.

            When testing out on my AMD RX6600M, this is my way
            of handling OOM errors.
        """
        cache = KVCache(
            num_layers = len(weights.layers),
            num_kv_heads = config.num_key_value_heads,
            head_dim = config.head_dim,
            max_seq_len = min(config.max_position_embeddings, 4096),
        )

    cos = rope_cos if rope_cos is not None else weights.rope_cos
    sin = rope_sin if rope_sin is not None else weights.rope_sin

    for layer_idx, layer in enumerate(weights.layers):
        builder = TransformerBlockBuilder(
            cos, sin,
            layer, layer_idx, position_offset,
        )
        x = _transformer_block(x, config, cache, builder)

        if isinstance(position_offset, int):
            x = x.realize(cache.keys[layer_idx], cache.values[layer_idx])

    if weights.final_norm is not None:
        x = rms_norm(x, weights.final_norm, config.rms_norm_eps)

    if weights.lm_head is not None:
        # Last rank (or single-rank): project to vocab space and return logits.
        logits = apply_lm_head(x, weights.lm_head)
        return logits, cache

    # Middle pipeline rank: return hidden state instead of logits.
    # Caller (pipeline transport) will ship this to the next rank.
    return x, cache

def _transformer_block(
    x: Tensor,
    config: ModelConfig,
    cache: KVCache,
    builder: TransformerBlockBuilder,
) -> Tensor:
    residual = x
    layer = builder.layer
    x = rms_norm(x, layer.input_norm, config.rms_norm_eps)

    match config.architecture_spec.attention_type:
        case "grouped_query" | "multi_head":
            x = grouped_query_attention(
                x, qkv_proj = layer.qkv_proj,
                o_proj = layer.o_proj,
                cos_freqs = builder.cos_freqs,
                sin_freqs = builder.sin_freqs,
                cache = cache, layer_idx = builder.idx,
                position_offset = builder.offset,
                cache_position = builder.offset,
                num_heads = config.num_attention_heads,
                num_kv_heads = config.num_key_value_heads,
                head_dim = config.head_dim,
                q_norm = layer.q_norm,
                k_norm = layer.k_norm,
                rms_norm_eps = config.rms_norm_eps,
            )
        case "multi_latent":
            raise NotImplementedError(
                "MLA attention: not yet been implemented"
            )

    x = x + residual
    residual = x

    x = rms_norm(x, layer.post_attn_norm, config.rms_norm_eps)
    match config.architecture_spec.mlp_type:
        case "swiglu":
            x = swiglu_mlp(
                x, layer.gate_up_proj, layer.down_proj
            )
        case "moe_top_k":
            raise NotImplementedError(
                "MoE MLP: not yet implemented"
            )

    x = x + residual
    return x

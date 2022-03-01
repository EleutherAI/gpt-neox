"""Transformer."""

import math
import torch
import torch.nn.functional as F
import torch.nn as nn

from .norms import get_norm
from megatron import mpu
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.activations import get_activation
from megatron.model.utils import exists, get_fusion_type
from megatron.model.positional_embeddings import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_torch,
    AliBi,
)
from megatron.model.fused_bias_dropout import (
    get_bias_dropout_add,
    bias_dropout_add_fused_train,
    bias_dropout_add_fused_inference,
)
from megatron.model.utils import configure_sparse_attention
from torchtyping import TensorType

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def gpt2_attention_mask_func(attention_scores, ltor_mask):
    attention_scores.masked_fill_(ltor_mask, -10000.0)
    return attention_scores


class ParallelMLP(nn.Module):
    def __init__(self, neox_args, parallel_output=False):
        super().__init__()
        self.neox_args = neox_args
        self.activation_func = get_activation(neox_args)

        self.dense_h_to_4h = nn.Linear(
            in_features=neox_args.hidden_size, out_features=4 * neox_args.hidden_size
        )
        self.dense_4h_to_h = nn.Linear(
            in_features=4 * neox_args.hidden_size, out_features=neox_args.hidden_size
        )

    def forward(
        self, hidden_states: TensorType["seq", "batch", "hidden_size"]
    ) -> TensorType["seq", "batch", "hidden_size"]:
        # [batch_size, seq_len, 4 * hidden_size]
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.activation_func(hidden_states)

        # [batch_size, seq_len, hidden_size]
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


class ParallelLinear(nn.Module):
    def __init__(self, neox_args, parallel_output=True):
        super().__init__()
        self.neox_args = neox_args
        self.final_linear = nn.Linear(
            in_features=neox_args.hidden_size,
            out_features=neox_args.padded_vocab_size,
            bias=False,
        )

    def forward(
        self, hidden_states: TensorType["seq", "batch", "hidden_size"]
    ) -> TensorType["seq", "batch", "vocab_size"]:
        return self.final_linear(hidden_states)


class ParallelSelfAttention(nn.Module):
    def __init__(
        self,
        neox_args,
        layer_number,
        parallel_output=False,
        use_cache=False,
        attention_mask_func=gpt2_attention_mask_func,
    ):
        super().__init__()
        self.neox_args = neox_args
        self.attention_mask_func = attention_mask_func
        self.layer_number = layer_number
        self.num_heads = neox_args.num_attention_heads
        self.use_cache = use_cache
        self.hidden_size_per_attention_head = neox_args.hidden_size // self.num_heads

        # q/k/v projection
        self.query_key_value = nn.Linear(
            in_features=neox_args.hidden_size,
            out_features=3 * neox_args.hidden_size,
            bias=False,
        )

        # scale
        coeff, self.norm_factor = self._init_scale()

        # positional embeddings
        self._init_pos_embeddings()

        # attention
        self.attention_type = neox_args.attention_config[layer_number]
        if self.attention_type != "global":
            raise NotImplementedError("Sparse attention not implemented")
        else:
            self.scale_mask_softmax = FusedScaleMaskSoftmax(
                input_in_fp16=self.neox_args.precision == "fp16",
                input_in_bf16=self.neox_args.precision == "bfloat16",
                fusion_type=get_fusion_type(neox_args),
                mask_func=self.attention_mask_func,
                softmax_in_fp32=self.attention_softmax_in_fp32,
                scale=coeff,
            )

            # Dropout
            self.attention_dropout = nn.Dropout(neox_args.attention_dropout)

        # Output.
        self.dense = nn.Linear(
            in_features=self.neox_args.hidden_size, out_features=neox_args.hidden_size
        )

    def _init_pos_embeddings(self):
        self.alibi_emb = None
        self.rotary_emb = None
        if self.neox_args.pos_emb == "alibi":
            raise NotImplementedError("alibi not implemented")
        elif self.neox_args.pos_emb == "rotary":
            assert self.neox_args.rotary_pct < 1.0
            if self.neox_args.rotary_pct == 1:
                self.rotary_ndims = None
            else:
                assert self.neox_args.rotary_pct < 1
                self.rotary_ndims = int(
                    self.hidden_size_per_attention_head * self.neox_args.rotary_pct
                )
            dim = self.rotary_ndims or self.hidden_size_per_attention_head
            self.rotary_emb = RotaryEmbedding(
                dim, precision=self.neox_args.params_dtype
            )

    def _init_scale(self):
        coeff = None
        norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.neox_args.apply_query_key_layer_scaling:
            coeff = max(1, self.layer_number)
            norm_factor *= coeff
        return coeff, norm_factor

    def _attn(self, q, k, v, layer_past, attention_mask):
        raise NotImplementedError("TODO")

    def forward(self, hidden_states, attention_mask, layer_past=None):
        # ==================================
        # qkv projection + split heads ([sq, b, h] --> [sq, b, (np * 3 * hn)])
        # ==================================

        qkv = self.query_key_value(hidden_states)
        q, k, v = torch.split(qkv, self.hidden_size_per_attention_head, dim=-1)

        # ==================================
        # rotary embedding
        # ==================================

        if self.rotary_emb is not None:
            q, k = self.rotary_emb(
                q,
                k,
                v,
                layer_past=layer_past,
                rotary_ndims=self.rotary_ndims,
                seq_dim=0,
            )

        # ==================================
        # Cache key and value for inference
        # ==================================

        if layer_past is not None and layer_past.numel() > 0:
            past_key, past_value = layer_past
            k = torch.cat((past_key.type_as(k), k), dim=0)
            v = torch.cat((past_value.type_as(v), v), dim=0)

        if self.use_cache:
            present = torch.stack((k, v))

        # ==================================
        # self-attention
        # ==================================

        if self.attention_type == "global":
            context_layer = self._attn(
                q, k, v, layer_past=layer_past, attention_mask=attention_mask
            )
        else:
            raise NotImplementedError("Sparse attention not implemented")

        # ==================================
        # Merge attention heads
        # ==================================
        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size_per_partition,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================
        output = self.dense(context_layer)

        if self.use_cache:
            output = (output, present)

        return output

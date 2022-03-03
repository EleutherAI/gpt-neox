#
# Copyright 2021 Biderman et al. This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer."""

import math

import torch
import torch.nn as nn

from .activations import get_activation
from .norms import get_norm
from .positional_embeddins import (
    RotaryEmbedding,
    apply_rotary_pos_emb_torch,
    apply_rotary_pos_emb,
)
from .softmax import FusedScaleMaskSoftmax

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
        attention_mask_func: a function that takes `unmasked-attention-scores`
            with size [b, np, s, s] and an `attention-mask` and will apply
            the masking. The function should return a masked score of the
            same size [b, np, s, s].
               masked-attention-scores = attention_mask_func(
                                     unmasked-attention-scores, attention-mask)
"""


class GPTNeoXMLP(nn.Module):
    """MLP.
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(
        self,
        hidden_size,
        init_method,
        output_layer_init_method,
        activation_type="gelu",
        onnx_safe=False,
        fused_gelu=True,
    ):
        super().__init__()

        self.activation_type = activation_type
        self.activation_func = get_activation(
            activation_type, onnx_safe=onnx_safe, fused_gelu=fused_gelu
        )

        # auto scale so geglu has equal parameters
        ff_mult = 4 * 2 / 3 if self.activation_type == "geglu" else 4
        ff_dim = (
            int(ff_mult * hidden_size) * 2
            if self.activation_type == "geglu"
            else ff_mult * hidden_size
        )
        self.dense_h_to_4h = nn.Linear(hidden_size, ff_dim)
        init_method(self.dense_h_to_4h.weight)
        ff_dim_in = ff_dim // 2 if self.activation_type == "geglu" else ff_dim

        # Project back to h.
        self.dense_4h_to_h = nn.Linear(ff_dim_in, hidden_size)
        output_layer_init_method(self.dense_4h_to_h)

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.activation_func(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


class GPTNeoXSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        rpe,
        rotary,
        use_cache,
        precision,
        apply_query_key_layer_scaling,
        attention_softmax_in_fp32,
        pos_emb,
        params_dtype,
        attention_dropout,
        rotary_pct=1,
        rotary_emb_base=10000,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_mask_func = attention_mask_func
        self.use_cache = use_cache
        self.fp16 = precision == "fp16"
        self.bf16 = precision == "bfloat16"
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.pos_emb = pos_emb
        self.params_dtype = params_dtype
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        self.query_key_value = nn.Linear(hidden_size, 3 * hidden_size)
        init_method(self.query_key_value.weight)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = max(1, self.layer_number)
            self.norm_factor *= coeff

        self.rpe = rpe
        self.rotary_pct = rotary_pct
        if rotary:
            if self.rotary_pct == 1:
                self.rotary_ndims = None
            else:
                assert rotary_pct < 1
                self.rotary_ndims = int(self.hidden_size * self.rotary_pct)
            dim = (
                self.rotary_ndims
                if self.rotary_ndims is not None
                else self.hidden_size_per_attention_head
            )
            self.rotary_emb = RotaryEmbedding(
                dim, rotary_emb_base, precision=params_dtype
            )
        else:
            self.rotary_emb = None
            self.rotary_ndims = None

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            input_in_fp16=self.fp16,
            input_in_bf16=self.bf16,
            mask_func=self.attention_mask_func,
            softmax_in_fp32=self.attention_softmax_in_fp32,
            scale=coeff,
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = nn.Dropout(attention_dropout)

        # Output.
        self.dense = nn.Linear(hidden_size, hidden_size)
        output_layer_init_method(self.dense.weight)

    def attention(self, query_layer, key_layer, value_layer, attention_mask):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(
            output_size[2], output_size[0] * output_size[1], -1
        )
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocating result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device(),
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if self.use_cache:
            with torch.no_grad():
                attention_mask = attention_mask[
                    ..., : attention_scores.size(3), : attention_scores.size(3)
                ]

        # ===========================
        # Attention probs and dropout
        # ===========================

        if self.rpe is not None:
            rpe = self.rpe(query_layer.size(0), key_layer.size(0))
            attention_scores += rpe  # [1, np, sq, sk]

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)
        attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(
            value_layer.size(0), output_size[0] * output_size[1], -1
        )

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], -1
        )

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        return context_layer

    def forward(self, hidden_states, attention_mask, layer_past=None):
        # hidden_states: [sq, b, h]
        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        query_layer, key_layer, value_layer = torch.chunk(mixed_x_layer, 3, dim=-1)

        if self.rotary_emb is not None:
            if self.rotary_ndims is not None:
                # partial rotary
                query_rot, query_pass = (
                    query_layer[..., : self.rotary_ndims],
                    query_layer[..., self.rotary_ndims :],
                )
                key_rot, key_pass = (
                    key_layer[..., : self.rotary_ndims],
                    key_layer[..., self.rotary_ndims :],
                )
            else:
                # full rotary
                query_rot, key_rot = query_layer, key_layer
            apply_rotary_fn = (
                apply_rotary_pos_emb_torch if self.bf16 else apply_rotary_pos_emb
            )

            seq_len = key_layer.shape[0]
            offset = 0
            if layer_past is not None and layer_past.numel() > 0:
                offset = layer_past[0].shape[0]
                seq_len += offset
            cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)
            query_layer, key_layer = apply_rotary_fn(
                query_rot, key_rot, cos, sin, offset=offset
            )

            if self.rotary_ndims is not None:
                query_layer = torch.cat((query_layer, query_pass), dim=-1)
                key_layer = torch.cat((key_layer, key_pass), dim=-1)

        # ==================================
        # Cache key and value for inference
        # ==================================

        if layer_past is not None and layer_past.numel() > 0:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
            value_layer = torch.cat(
                (past_value.type_as(value_layer), value_layer), dim=0
            )

        if self.use_cache:
            present = torch.stack((key_layer, value_layer))

        context_layer = self.attention(
            query_layer, key_layer, value_layer, attention_mask
        )

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
            output = [output, present]

        return output


class GPTNeoXLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        layer_number,
        norm_type,
        norm_eps,
        hidden_drop_prob,
        num_attention_heads,
        precision,
        apply_query_key_layer_scaling,
        attention_softmax_in_fp32,
        pos_emb,
        params_dtype,
        attention_dropout,
        rotary_pct=1,
        rotary_emb_base=10000,
        rpe=None,
        rotary=False,
        use_cache=False,
    ):
        super().__init__()
        self.layer_number = layer_number
        self.use_cache = use_cache
        self.layer_past = None  # used to cache k/v pairs in inference

        norm = get_norm(norm_type)
        self.input_layernorm = norm(hidden_size, eps=norm_eps)

        self.attention = GPTNeoXSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_mask_func=attention_mask_func,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            rpe=rpe,
            rotary=rotary,
            use_cache=use_cache,
            precision=precision,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            attention_softmax_in_fp32=attention_softmax_in_fp32,
            pos_emb=pos_emb,
            params_dtype=params_dtype,
            attention_dropout=attention_dropout,
            rotary_pct=rotary_pct,
            rotary_emb_base=rotary_emb_base,
        )

        # Layernorm on the output of the attention layer.
        self.post_attention_layernorm = norm(hidden_size, eps=norm_eps)

        self.mlp = GPTNeoXMLP(
            hidden_size=hidden_size,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
        )

        self.hidden_dropout = nn.Dropout(hidden_drop_prob)

    def forward(self, x, attention_mask, layer_past=None):
        # x: [b, s, h]
        layer_past = layer_past if layer_past is not None else self.layer_past

        # pseudocode:
        # x = x + attn(ln1(x))
        # x = x + mlp(ln2(x))
        residual = x

        # x = x + attn(ln1(x))
        attention_output = self.attention(
            self.input_layernorm(x), attention_mask, layer_past=layer_past
        )
        if self.use_cache:
            attention_output, presents = attention_output
            self.layer_past = presents

        attention_output = self.hidden_dropout(attention_output)
        attention_output += residual

        # output = x + mlp(ln2(x))
        mlp_output = self.mlp(self.post_attention_layernorm(attention_output))
        mlp_output = self.hidden_dropout(mlp_output)
        mlp_output += residual

        return mlp_output

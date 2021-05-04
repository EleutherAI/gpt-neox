# coding=utf-8
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
import torch.nn.functional as F

from .norms import LayerNorm, RMSNorm, ScaleNorm
from megatron import mpu
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.utils import openai_gelu, erf_gelu, exists
from megatron.model.positional_embeddings import RotaryEmbedding, apply_rotary_pos_emb
from megatron.model.fused_bias_dropout import get_bias_dropout_add, bias_dropout_add_fused_train, \
    bias_dropout_add_fused_inference
from megatron.model.utils import configure_sparse_attention

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

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
        attention_mask_func: a function that takes `unmaksed-attention-scores`
            with size [b, np, s, s] and an `attention-mask` and will apply
            the masking. The function should return a masked score of the
            same size [b, np, s, s].
               masked-attention-scores = attention_mask_func(
                                     unmaksed-attention-scores, attention-mask)
"""


class GEGLU(torch.nn.Module):

    def __init__(self, neox_args):
        super(GEGLU, self).__init__()

        self.bias_gelu_fusion = neox_args.bias_gelu_fusion
        self.activation_func = F.gelu
        if neox_args.openai_gelu:
            self.activation_func = openai_gelu
        elif neox_args.onnx_safe:
            self.activation_func = erf_gelu

    def forward(self, x, bias=None):
        x, gate = x.chunk(2, dim=-1)
        if bias is not None:
            bias_1, bias_2 = bias.chunk(2, dim=-1)
            x = x + bias_1
        else:
            bias_1 = bias_2 = 0
        if self.bias_gelu_fusion:
            intermediate_parallel = \
                bias_gelu_impl(gate, bias_2)
        else:
            intermediate_parallel = \
                self.activation_func(gate + bias_2)
        return intermediate_parallel * x


class ParallelMLP(torch.nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(self, neox_args, init_method, output_layer_init_method):
        super(ParallelMLP, self).__init__()

        if neox_args.geglu:
            self.activation_type = "geglu"
            mult = 8
            self.activation_func = GEGLU(neox_args=neox_args)
        else:
            self.activation_type = "gelu"
            mult = 4
            self.bias_gelu_fusion = neox_args.bias_gelu_fusion
            self.activation_func = F.gelu
            if neox_args.openai_gelu:
                self.activation_func = openai_gelu
            elif neox_args.onnx_safe:
                self.activation_func = erf_gelu

        # Project to 4h.
        self.dense_h_to_4h = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=mult * neox_args.hidden_size,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True
        )

        # Project back to h.
        self.dense_4h_to_h = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=4 * neox_args.hidden_size,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True)

    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if self.activation_type == "gelu":
            if self.bias_gelu_fusion:
                intermediate_parallel = \
                    bias_gelu_impl(intermediate_parallel, bias_parallel)
            else:
                intermediate_parallel = \
                    self.activation_func(intermediate_parallel + bias_parallel)
        elif self.activation_type == "geglu":
            intermediate_parallel = \
                self.activation_func(intermediate_parallel)
        else:
            raise ValueError(f'Activation type {self.activation_type} not recognized')

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


class ParallelLinear(torch.nn.Module):
    """
    A Parallel Linear Layer transforming the transformer outputs from hidden_size -> vocab_size
    """

    def __init__(self, neox_args, parallel_output=True, init_method=torch.nn.init.xavier_normal_):
        super(ParallelLinear, self).__init__()
        self.final_linear = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=neox_args.padded_vocab_size,
            bias=False,
            input_is_parallel=False,
            init_method=init_method,
            parallel_output=parallel_output,
            skip_bias_add=False)

    def forward(self, hidden_states):
        return self.final_linear(hidden_states)


class ParallelSelfAttention(torch.nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, neox_args, attention_mask_func, init_method,
                 output_layer_init_method, layer_number,
                 rpe=None, rotary=False, get_key_value=False):
        super(ParallelSelfAttention, self).__init__()

        self.fp16 = neox_args.precision == "fp16"
        self.attention_mask_func = attention_mask_func
        self.apply_query_key_layer_scaling = neox_args.apply_query_key_layer_scaling
        self.get_key_value = get_key_value
        self.attention_softmax_in_fp32 = neox_args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = layer_number
        # Per attention head and per partition values.
        world_size = mpu.get_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(neox_args.hidden_size,
                                                    world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            neox_args.hidden_size, neox_args.num_attention_heads)
        self.num_attention_heads_per_partition = mpu.divide(
            neox_args.num_attention_heads, world_size)

        # Strided linear layer.
        self.query_key_value = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=3 * neox_args.hidden_size,
            gather_output=False,
            init_method=init_method)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = max(1, self.layer_number)
            self.norm_factor *= coeff

        self.rpe = rpe

        if rotary:
            if neox_args.rotary_pct == 1:
                self.rotary_ndims = None
            else:
                assert neox_args.rotary_pct < 1
                self.rotary_ndims = int(self.hidden_size_per_attention_head * neox_args.rotary_pct)
            dim = self.rotary_ndims if self.rotary_ndims is not None else self.hidden_size_per_attention_head
            self.rotary_emb = RotaryEmbedding(dim, base=neox_args.rotary_emb_base)
        else:
            self.rotary_emb = None

        self.attention_type = neox_args.attention_config[layer_number]
        self.sparse = self.attention_type != 'global'
        if self.sparse:
            self.sparse_attn = configure_sparse_attention(neox_args, self.attention_type,
                                                          self.num_attention_heads_per_partition,
                                                          mpu=mpu)
        else:
            self.scale_mask_softmax = FusedScaleMaskSoftmax(
                self.fp16,
                neox_args.scaled_upper_triang_masked_softmax_fusion,
                neox_args.scaled_masked_softmax_fusion,
                self.attention_mask_func,
                self.attention_softmax_in_fp32,
                coeff)

            # Dropout. Note that for a single iteration, this layer will generate
            # different outputs on different number of parallel partitions but
            # on average it should not be partition dependent.
            self.attention_dropout = torch.nn.Dropout(neox_args.attention_dropout)

        # Output.
        self.dense = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True)

    def attention(self, query_layer, key_layer, value_layer, layer_past, attention_mask):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocating result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(output_size[0] * output_size[1], output_size[2], output_size[3],
                                    dtype=query_layer.dtype, device=torch.cuda.current_device())

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(matmul_result,
                                      query_layer.transpose(0, 1),  # [b * np, sq, hn]
                                      key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                                      beta=0.0, alpha=(1.0 / self.norm_factor))

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if self.get_key_value:
            with torch.no_grad():
                if layer_past is not None and layer_past.numel() > 0:
                    attention_mask = attention_mask[
                                     ...,
                                     attention_scores.size(3) - 1,
                                     :attention_scores.size(3)].unsqueeze(2)
                else:
                    attention_mask = attention_mask[
                                     ...,
                                     :attention_scores.size(3),
                                     :attention_scores.size(3)]

        # ===========================
        # Attention probs and dropout
        # ===========================

        if exists(self.rpe):
            rpe = self.rpe(query_layer.size(0), key_layer.size(0))
            attention_scores += rpe  # [1, np, sq, sk]

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        return context_layer

    def sparse_attention(self, query_layer, key_layer, value_layer, attention_mask):
        # TODO: sparse attn dropout?
        # TODO: pad to block size
        # shape of q/k/v is [sq, b, np, hn] and needs to be transposed to [b, np, sq, hn]
        query_layer, key_layer, value_layer = map(lambda t: t.permute(1, 2, 0, 3).contiguous(),
                                                  (query_layer, key_layer,
                                                   value_layer))
        # output shape [b, np(heads), sq, hn]
        attn_mask = attention_mask.to(query_layer.dtype) * -10000
        if exists(self.rpe):
            rpe = self.rpe(query_layer.size(0), key_layer.size(0))
        else:
            rpe = None
        return self.sparse_attn(query_layer, key_layer, value_layer, attn_mask=attn_mask, rpe=rpe)

    def forward(self, hidden_states, attention_mask, layer_past=None):

        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (self.num_attention_heads_per_partition,
                                                        3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer, key_layer, value_layer) = mpu.split_tensor_along_last_dim(mixed_x_layer, 3)

        if exists(self.rotary_emb):
            if exists(self.rotary_ndims):
                # partial rotary
                query_rot, query_pass = query_layer[..., :self.rotary_ndims], query_layer[..., self.rotary_ndims:]
                key_rot, key_pass = key_layer[..., :self.rotary_ndims], key_layer[..., self.rotary_ndims:]
                cos, sin = self.rotary_emb(query_rot, seq_dim=0)
            else:
                # full rotary
                cos, sin = self.rotary_emb(query_layer, seq_dim=0)
                query_rot, key_rot = query_layer, key_layer

            query_layer, key_layer = apply_rotary_pos_emb(query_rot, key_rot, cos, sin)

            if exists(self.rotary_ndims):
                query_layer = torch.cat((query_layer, query_pass), dim=-1)
                key_layer = torch.cat((key_layer, key_pass), dim=-1)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        if layer_past is not None and layer_past.numel() > 0:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer),
                                   key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer),
                                     value_layer), dim=0)
        if self.get_key_value:
            present = torch.stack((key_layer, value_layer))

        if not self.sparse:
            context_layer = self.attention(query_layer, key_layer, value_layer, layer_past, attention_mask)
        else:
            context_layer = self.sparse_attention(query_layer, key_layer, value_layer, attention_mask)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
                                  (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        if self.get_key_value:
            output = [output, present]

        return output, bias


class ParallelTransformerLayer(torch.nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(self, neox_args, attention_mask_func, init_method,
                 output_layer_init_method, layer_number, rpe=None, rotary=False, get_key_value=False):

        super(ParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm = neox_args.apply_residual_connection_post_layernorm

        if neox_args.norm == "rmsnorm":
            norm = RMSNorm
            eps = neox_args.rms_norm_epsilon
        elif neox_args.norm == "layernorm":
            eps = neox_args.layernorm_epsilon
            norm = LayerNorm
        elif neox_args.norm == "scalenorm":
            eps = neox_args.scalenorm_epsilon
            norm = ScaleNorm

        # Layernorm on the input data.
        self.input_layernorm = norm(neox_args.hidden_size, eps=eps)
        self.get_key_value = get_key_value

        # Self attention.
        self.attention = ParallelSelfAttention(
            neox_args=neox_args,
            attention_mask_func=attention_mask_func,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            layer_number=layer_number,
            rpe=rpe,
            get_key_value=self.get_key_value,
            rotary=rotary
        )

        self.hidden_dropout = neox_args.hidden_dropout
        self.bias_dropout_fusion = neox_args.bias_dropout_fusion

        # Layernorm on the input data.
        self.post_attention_layernorm = norm(
            neox_args.hidden_size,
            eps=eps)

        # MLP
        self.mlp = ParallelMLP(
            neox_args=neox_args,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method
        )

    def forward(self, hidden_states, attention_mask, layer_past=None):
        # hidden_states: [b, s, h]

        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, attention_bias = \
            self.attention(layernorm_output,
                           attention_mask,
                           layer_past=layer_past)

        if self.get_key_value:
            attention_output, presents = attention_output

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # jit scripting for a nn.module (with dropout) is not 
        # trigerring the fusion kernel. For now, we use two 
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.
        if self.bias_dropout_fusion:
            if self.training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            bias_dropout_add_func = get_bias_dropout_add(self.training)

        # re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            layernorm_input = bias_dropout_add_func(
                attention_output,
                attention_bias.expand_as(residual),
                residual,
                self.hidden_dropout)

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output, mlp_bias = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        # re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            output = bias_dropout_add_func(
                mlp_output,
                mlp_bias.expand_as(residual),
                residual,
                self.hidden_dropout)

        if self.get_key_value:
            output = [output, presents]

        return output


class ParallelTransformerLayerPipe(ParallelTransformerLayer):
    """Extends ParallelTransformerLayer to forward attention_mask through the pipeline. """

    def forward(self, args):
        in_inference = len(args) == 4  # length of the args in inference == 4
        in_train = len(args) == 2  # length of the args in training == 2
        if in_train:
            hidden_states, attention_mask = args
            # we are returning just [hidden_states, mask]
            return super().forward(hidden_states, attention_mask), attention_mask
        elif in_inference:
            # we are in inference
            hidden_states, layer_past, presents, attention_mask = args
            past = torch.Tensor()
            if layer_past is not None and layer_past.numel() > 0:
                past = layer_past[self.layer_number]
            outputs = super().forward(hidden_states, attention_mask, layer_past=past)

            if self.get_key_value:
                # outputs = [hidden_states, present]
                hidden_states, present = outputs
                if presents.numel() == 0:
                    presents = present.unsqueeze(dim=0)
                else:
                    presents = torch.cat((presents, present.unsqueeze(dim=0)))
            else:
                hidden_states = outputs
            return hidden_states, layer_past, presents, attention_mask
        else:
            raise ValueError(
                f'In layer {self.layer_number} - Incorrect number of arguments ({len(args)}) for {self.__class__.__name__}')


class ParallelLinearPipe(ParallelLinear):
    """Another helper class to pass presents through to the output when doing inference with a Pipe Parallel model"""

    def forward(self, args):
        if not isinstance(args, tuple):
            # in training, args = hidden_state (tensor, so we check if object isn't a tuple and pass through here)
            hidden_state = args
            logits, bias = super().forward(hidden_state)
            return logits
        elif len(args) == 2:
            # we are in inference, so input is (hidden_states, presents)
            hidden_state, presents = args
            logits, bias = super().forward(hidden_state)
            return logits, presents
        else:
            raise ValueError(f'Incorrect number of arguments for {self.__class__.__name__}')


class NormPipe(torch.nn.Module):
    """Just a helper class to pass presents through to the output when doing inference with a Pipe Parallel model"""

    def __init__(self, norm_class, hidden_size, eps):
        super().__init__()
        self.norm = norm_class(hidden_size, eps=eps)

    def forward(self, args):
        if not isinstance(args, tuple):
            # in training, args = hidden_state (tensor, so we check if object isn't a tuple and pass through here)
            hidden_state = args
            return self.norm(hidden_state)
        elif len(args) == 2:
            # in inference, args will be (hidden_state, presents)
            hidden_state, presents = args
            hidden_state = self.norm(hidden_state)
            return hidden_state, presents
        else:
            raise ValueError(f'Incorrect number of arguments for {self.__class__.__name__}')


def parallel_lm_logits(input_, word_embeddings_weight, parallel_output,
                       bias=None):
    """LM logits using word embedding weights."""
    # Parallel logits.
    input_parallel = mpu.copy_to_model_parallel_region(input_)

    # Matrix multiply.
    if bias is None:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight)
    else:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight, bias)

    # Gather if needed.
    if parallel_output:
        return logits_parallel

    return mpu.gather_from_model_parallel_region(logits_parallel)

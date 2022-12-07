# Copyright (c) 2021, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
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

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.activations import get_activation
from megatron.model.norms import get_norm
from megatron.model.utils import get_fusion_type

from megatron import mpu


class TinyAttention(nn.Module):
    def __init__(self, neox_args, d_attn, d_ff, mask_fn):
        super().__init__()
        self.proj_qkv = nn.Linear(d_ff * 2, 3 * d_attn)
        self.scale = d_attn**-0.5
        self.proj_ffn = nn.Linear(d_attn, d_ff)
        self.softmax = FusedScaleMaskSoftmax(
            input_in_fp16=neox_args.precision == "fp16",
            input_in_bf16=neox_args.precision == "bfloat16",
            fusion_type=get_fusion_type(neox_args),
            mask_func=mask_fn,
            softmax_in_fp32=neox_args.attention_softmax_in_fp32,
            scale=None,
        )

    def forward(self, x, attention_mask):
        q, k, v = torch.chunk(self.proj_qkv(x), 3, dim=-1)
        w = torch.einsum("bnd,bmd->bnm", q, k).unsqueeze(1) * self.scale
        a = self.softmax(
            w, mask=attention_mask[..., : w.size(-2), : w.size(-1)]
        ).squeeze(1)
        x = torch.einsum("bnm,bmd->bnd", a, v)
        return self.proj_ffn(x)


class SpatialGatingUnit(nn.Module):
    def __init__(self, neox_args, d_ff, d_attn=None, causal=True, mask_fn=None):
        super().__init__()
        self.causal = causal
        self.use_attn = d_attn is not None

        norm, eps = get_norm(neox_args)
        self.norm = norm(d_ff, eps=eps)
        self.proj = nn.Linear(neox_args.seq_length, neox_args.seq_length)
        if self.use_attn:
            assert mask_fn is not None
            self.attn = TinyAttention(
                neox_args=neox_args, d_attn=d_attn, d_ff=d_ff, mask_fn=mask_fn
            )
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 1.0)

    def forward(self, x, attention_mask):
        device, n = x.device, x.shape[1]
        x = x.transpose(0, 1)  # [s, b, d] -> [b, s, d]

        res, gate = x.chunk(2, dim=-1)  # split along dim
        gate = self.norm(gate)

        weight, bias = self.proj.weight, self.proj.bias
        if self.causal:
            weight, bias = weight[:n, :n], bias[:n]
            mask = torch.ones(weight.shape[:2], device=device).triu_(1).bool()
            weight = weight.masked_fill(mask, 0.0)

        gate = F.linear(gate.transpose(2, 1), weight, self.proj.bias).transpose(2, 1)

        if self.use_attn:
            gate = gate + self.attn(x, attention_mask)

        return (gate * res).transpose(0, 1)  # [b, s, d] -> [s, b, d]


class GMLPBlock(nn.Module):
    def __init__(
        self,
        neox_args,
        init_method,
        output_layer_init_method,
        layer_number,
        ff_mult=4,
        mask_fn=None,
    ):
        super().__init__()
        self.layer_number = layer_number

        ff_dim = neox_args.hidden_size * ff_mult
        norm, eps = get_norm(neox_args)
        self.norm = norm(neox_args.hidden_size, eps=eps)
        self.input_linear = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=ff_dim * 2,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
        )
        self.activation_func = get_activation(neox_args)
        ff_dim_parallel = mpu.divide(ff_dim, mpu.get_model_parallel_world_size())
        if neox_args.attention_config[layer_number] == "amlp":
            d_attn = neox_args.gmlp_attn_dim
        else:
            d_attn = None
        self.sgu = SpatialGatingUnit(
            neox_args, ff_dim_parallel, d_attn, causal=True, mask_fn=mask_fn
        )
        self.output_linear = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=ff_dim,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
        )

    def forward(self, args):
        assert len(args) == 2, "GMLPBlock expects 2 arguments"
        x, attention_mask = args
        x = self.norm(x)
        x, _ = self.input_linear(x)
        x = self.activation_func(x)
        x = self.sgu(x, attention_mask)
        x, _ = self.output_linear(x)
        return x, attention_mask

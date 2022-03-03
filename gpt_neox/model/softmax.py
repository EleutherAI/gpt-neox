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

import torch
import torch.nn as nn

from gpt_neox.csrc.binders import get_scaled_upper_triang_masked_softmax_cuda


class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, scale):
        scale_t = torch.tensor([scale])
        softmax_results = get_scaled_upper_triang_masked_softmax_cuda().forward(
            inputs, scale_t[0]
        )
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        softmax_results, scale_t = ctx.saved_tensors

        input_grads = get_scaled_upper_triang_masked_softmax_cuda().backward(
            output_grads, softmax_results, scale_t[0]
        )
        return input_grads, None


class FusedScaleMaskSoftmax(nn.Module):
    """
    fused operation: scaling + mask + softmax

    Arguments:
        input_in_fp16: flag to indicate if input in fp16 data format.
        input_in_bf16: flag to indicate if input in bf16 data format.
        fusion_type: type of fusion to perform, should be either upper_triang, general or none. None will perform a regular torch softmax.
        mask_func: mask function to be applied.
        softmax_in_fp32: if true, softmax in performed at fp32 precision.
        scale: scaling factor used in input tensor scaling.
    """

    def __init__(
        self,
        input_in_fp16,
        input_in_bf16,
        mask_func,
        softmax_in_fp32,
        scale,
    ):
        super().__init__()
        self.input_in_fp16 = input_in_fp16
        self.input_in_bf16 = input_in_bf16
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale

        assert (
            self.scale is None or softmax_in_fp32
        ), "softmax should be in fp32 when scaled"

    def forward(self, input, mask):
        # [b, np, sq, sk]
        assert input.dim() == 4
        if self.is_kernel_available(mask, *input.size()):
            return self.forward_fused_softmax(input)
        else:
            return self.forward_torch_softmax(input, mask)

    @staticmethod
    def is_kernel_available(dtype, bsz, np, sq, sk):
        if dtype != torch.half or sk > 2048 or sk <= 0:
            return False

        bsz_per_block = (
            get_scaled_upper_triang_masked_softmax_cuda().get_batch_per_block(
                sq, sk, bsz, np
            )
        )

        if sq == sk and (sk <= 64 or sk % 4 == 0) and (bsz * np) % bsz_per_block == 0:
            return True

        return False

    def forward_fused_softmax(self, input):
        b, np, sq, sk = input.size()
        scale = self.scale if self.scale is not None else 1.0
        assert sq == sk, "causal mask is only for self attention"

        # input is 3D tensor (attn_batches, sq, sk)
        input = input.view(-1, sq, sk)
        probs = ScaledUpperTriangMaskedSoftmax.apply(input, scale)
        return probs.view(b, np, sq, sk)

    def forward_torch_softmax(self, input, mask):
        if self.input_in_float16 and self.softmax_in_fp32:
            input = input.float()

        if self.scale is not None:
            input = input * self.scale
        mask_output = self.mask_func(input, mask) if mask is not None else input
        probs = torch.nn.Softmax(dim=-1)(mask_output)

        if self.input_in_float16 and self.softmax_in_fp32:
            if self.input_in_fp16:
                probs = probs.half()
            else:
                probs = probs.bfloat16()

        return probs

    @staticmethod
    def get_batch_per_block(sq, sk, b, np):
        return get_scaled_upper_triang_masked_softmax_cuda().get_batch_per_block(
            sq, sk, b, np
        )

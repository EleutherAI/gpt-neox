# Copyright (c) 2025, EleutherAI
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

import math
import os
from dataclasses import replace
from io import BytesIO

import torch
import torch.nn.functional as F
import torch.nn.init as init
from packaging.version import Version
from torch.nn.parameter import Parameter

from megatron.model.transformer import Gated_Activation
from megatron.model.activations import get_activation
from megatron.mpu.initialize import get_model_parallel_rank
from megatron.mpu.initialize import get_model_parallel_world_size
from megatron.mpu.initialize import get_tensor_model_parallel_group
from megatron.mpu.mappings import copy_to_model_parallel_region
from megatron.mpu.mappings import gather_from_model_parallel_region
from megatron.mpu.mappings import reduce_from_model_parallel_region
from megatron.mpu.mappings import scatter_to_model_parallel_region
from megatron.mpu.mappings import reduce_scatter_to_sequence_parallel_region
from megatron.mpu.mappings import gather_from_sequence_parallel_region
from megatron.mpu.layers import (
    _initialize_affine_weight_gpu,
    _initialize_affine_weight_cpu,
)
from megatron.mpu.random import get_cuda_rng_tracker
from megatron.mpu.utils import divide
from megatron.mpu.utils import VocabUtility
from megatron.model.positional_embeddings import RotaryEmbedding
from megatron import mpu

# https://github.com/NVIDIA/TransformerEngine/issues/405
os.environ["NVTE_TORCH_COMPILE"] = str(0)

try:
    import transformer_engine as te
except ImportError:
    raise ImportError(
        "Unable to import transformer-engine. Please refer to "
        "https://github.com/NVIDIA/TransformerEngine for installation instructions."
    )


class TERMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-8, **kwargs):
        """
            A conditional wrapper to initialize an instance of Transformer-Engine's
            `RMSNorm` based on input
        :param dim: model size
        :param eps:  epsilon value, default 1e-8
        """
        super(TERMSNorm, self).__init__()

        self.d = dim
        self.eps = eps
        self.norm = te.pytorch.RMSNorm(
            hidden_size=self.d,
            eps=self.eps,
            **kwargs,
        )

    def forward(self, x):
        return self.norm(x)


class TELayerNorm(torch.nn.Module):
    def __init__(self, dim, eps=1.0e-5, **kwargs):
        """
            A conditional wrapper to initialize an instance of Transformer-Engine's
            `LayerNorm` based on input
        :param dim: model size
        :param eps:  epsilon value, default 1.0e-5
        """
        super(TELayerNorm, self).__init__()

        self.d = dim
        self.eps = eps
        self.norm = te.pytorch.LayerNorm(
            hidden_size=self.d,
            eps=self.eps,
            **kwargs,
        )

    def forward(self, x):
        return self.norm(x)


class TELinear(te.pytorch.Linear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer.
    """

    def __init__(
        self,
        neox_args,
        input_size,
        output_size,
        bias=True,
        init_method=init.xavier_normal_,
        stride=1,
        skip_bias_add=False,
        seq_dim=0,
    ):
        self.input_size = input_size
        self.output_size = output_size

        self.skip_bias_add = skip_bias_add
        self.use_bias = bias

        self.sequence_parallel = neox_args.sequence_parallel
        self.seq_dim = seq_dim

        self.init_method = init_method
        self.stride = stride
        self.params_dtype = neox_args.params_dtype

        super(TELinear, self).__init__(
            in_features=self.input_size,
            out_features=self.output_size,
            bias=self.use_bias,
            init_method=self.init_method,
            get_rng_state_tracker=get_cuda_rng_tracker,
            device=torch.cuda.current_device(),
            return_bias=self.skip_bias_add,
            params_dtype=self.params_dtype,
        )

    def forward(self, inp, **kwargs):
        output = super(TELinear, self).forward(inp, **kwargs)

        if self.skip_bias_add:
            return output
        else:
            return output, None


class TELayerNormMLP(te.pytorch.LayerNormMLP):
    """
    Wrapper for the Transformer-Engine's `LayerNormMLP` layer that combines
    layernorm and followed by the MLP module, consisting of 2 successive
    linear transformations, separated by the GeLU activation.
    """

    def __init__(
        self,
        neox_args,
        init_method,
        output_layer_init_method,
        parallel_output=False,
        multiple_of=256,
        MOE=False,
        MoE_mp_size=1,
        bias=True,
    ):
        self.activation_type = neox_args.activation
        self.multiple_of = multiple_of
        self.bias = bias
        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method

        world_size = MoE_mp_size if MOE else get_model_parallel_world_size()
        self.world_size = world_size
        self.tp_group = get_tensor_model_parallel_group()
        self.sequence_parallel = neox_args.sequence_parallel
        self.seq_len = neox_args.seq_length
        self.micro_batch_size = neox_args.train_micro_batch_size_per_gpu
        self.params_dtype = neox_args.params_dtype
        self.set_parallel_mode = False
        if world_size > 1:
            self.set_parallel_mode = True

        if neox_args.intermediate_size:
            ffn_dim = neox_args.intermediate_size
        elif neox_args.expansion_factor:
            ffn_dim = int(neox_args.expansion_factor * neox_args.hidden_size)
        else:
            # 4h is default for ffn_dim
            ffn_dim = 4 * neox_args.hidden_size

        if neox_args.norm in ["layernorm", "te_layernorm"]:
            self.eps = 1.0e-5
            self.normalization = "LayerNorm"
        elif neox_args.norm in ["rmsnorm", "te_rmsnorm"]:
            self.eps = 1.0e-8
            self.normalization = "RMSNorm"
        else:
            raise ValueError(
                "Only LayerNorm and RMSNorm are supported with TransformerEngine"
            )

        if self.activation_type not in [
            "gelu",
            "geglu",
            "relu",
            "reglu",
            "squared_relu",
            "swiglu",
            "qgelu",
            "srelu",
        ]:
            raise ValueError(
                "Only gelu, geglu, relu, reglu, squared_relu, swiglu, qgelu, and srelu are supported with TransformerEngine"
            )

        super(TELayerNormMLP, self).__init__(
            hidden_size=neox_args.hidden_size,
            ffn_hidden_size=ffn_dim,
            eps=self.eps,
            bias=self.bias,
            normalization=self.normalization,
            activation=self.activation_type,
            init_method=self.init_method,
            output_layer_init_method=self.output_layer_init_method,
            device=torch.cuda.current_device(),
            set_parallel_mode=self.set_parallel_mode,
            sequence_parallel=self.sequence_parallel,
            tp_group=self.tp_group,
            tp_size=self.world_size,
            return_bias=True,
            params_dtype=self.params_dtype,
            seq_length=self.seq_len,
            get_rng_state_tracker=get_cuda_rng_tracker,
            micro_batch_size=self.micro_batch_size,
        )


class TEColumnParallelLinear(te.pytorch.Linear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `ColumnParallelLinear` layer.

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
    """

    def __init__(
        self,
        neox_args,
        input_size,
        output_size,
        bias=True,
        gather_output=True,
        init_method=init.xavier_normal_,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        MOE=False,
        MoE_mp_size=1,
        seq_dim=0,
    ):
        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = MoE_mp_size if MOE else get_model_parallel_world_size()
        self.world_size = world_size
        self.tp_group = get_tensor_model_parallel_group()
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add
        self.use_bias = bias

        self.sequence_parallel = neox_args.sequence_parallel
        self.seq_dim = seq_dim

        self.init_method = init_method
        self.stride = stride
        self.params_dtype = neox_args.params_dtype
        self.parallel_mode = "column"

        super(TEColumnParallelLinear, self).__init__(
            in_features=self.input_size,
            out_features=self.output_size,
            bias=self.use_bias,
            init_method=self.init_method,
            get_rng_state_tracker=get_cuda_rng_tracker,
            device=torch.cuda.current_device(),
            sequence_parallel=self.sequence_parallel,
            tp_group=self.tp_group,
            tp_size=self.world_size,
            parallel_mode=self.parallel_mode,
            return_bias=self.skip_bias_add and self.use_bias,
            params_dtype=self.params_dtype,
        )

    # Copied from Mup
    def forward(self, inp, **kwargs):
        output = super(TEColumnParallelLinear, self).forward(inp, **kwargs)
        if self.return_bias:
            return output
        return output, None


class TERowParallelLinear(te.pytorch.Linear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `RowParallelLinear` layer.

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
    """

    def __init__(
        self,
        neox_args,
        input_size,
        output_size,
        bias=True,
        input_is_parallel=False,
        init_method=init.xavier_normal_,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        MOE=False,
        MoE_mp_size=1,
        parallel_output=False,
    ):
        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        # Divide the weight matrix along the last dimension.
        world_size = MoE_mp_size if MOE else get_model_parallel_world_size()
        self.world_size = world_size
        self.tp_group = get_tensor_model_parallel_group()
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add
        self.use_bias = bias
        self.input_is_parallel = input_is_parallel
        self.sequence_parallel = neox_args.sequence_parallel

        self.init_method = init_method
        self.stride = stride
        self.params_dtype = neox_args.params_dtype
        self.parallel_mode = "row"

        super(TERowParallelLinear, self).__init__(
            in_features=self.input_size,
            out_features=self.output_size,
            bias=self.use_bias,
            init_method=self.init_method,
            get_rng_state_tracker=get_cuda_rng_tracker,
            device=torch.cuda.current_device(),
            sequence_parallel=self.sequence_parallel,
            tp_group=self.tp_group,
            tp_size=self.world_size,
            parallel_mode=self.parallel_mode,
            return_bias=self.skip_bias_add and self.use_bias,
            params_dtype=self.params_dtype,
        )

    # Copied from Mup
    def forward(self, inp, **kwargs):
        output = super(TERowParallelLinear, self).forward(inp, **kwargs)

        if self.return_bias:
            return output
        return output, None


class TEMultiheadAttention(te.pytorch.MultiheadAttention):
    """
    Wrapper for the Transformer-Engine's `MultiheadAttention` layer that also
    has "flash attention" enabled.
    """

    def __init__(
        self,
        neox_args,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        layer_number,
        rpe=None,
        rotary=False,
        use_cache=False,
        parallel_output=False,
    ):

        self.neox_args = neox_args
        self.attention_mask_func = attention_mask_func
        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method
        self.layer_number = layer_number + 1

        world_size = get_model_parallel_world_size()
        self.world_size = world_size
        self.tp_group = get_tensor_model_parallel_group()
        self.sequence_parallel = neox_args.sequence_parallel
        self.seq_len = neox_args.seq_length
        self.micro_batch_size = neox_args.train_micro_batch_size_per_gpu
        self.params_dtype = neox_args.params_dtype
        self.set_parallel_mode = False
        if world_size > 1:
            self.set_parallel_mode = True

        if neox_args.norm in ["layernorm", "te_layernorm"]:
            self.eps = 1.0e-5
            self.normalization = "LayerNorm"
        elif neox_args.norm in ["rmsnorm", "te_rmsnorm"]:
            self.eps = 1.0e-8
            self.normalization = "RMSNorm"

        if (
            not neox_args.num_kv_heads
            or neox_args.num_kv_heads == neox_args.num_attention_heads
        ):
            self.gqa = False
            self.num_kv_heads = None
        else:
            self.gqa = True
            self.num_kv_heads = neox_args.num_kv_heads

        # TE uses one bias flag for both the Q/K/V and output projections, so
        # NeoXArgs rejects "qkv" here and only "all" and "none" reach this point.
        use_bias = neox_args.use_bias_in_attn_linear == "all"

        super(TEMultiheadAttention, self).__init__(
            hidden_size=neox_args.hidden_size,
            num_attention_heads=neox_args.num_attention_heads,
            attention_dropout=neox_args.attention_dropout,
            layernorm_epsilon=self.eps,
            init_method=self.init_method,
            output_layer_init_method=self.output_layer_init_method,
            layer_number=self.layer_number,
            window_size=(
                (neox_args.sliding_window_width, 0)
                if neox_args.sliding_window_width is not None
                else None
            ),
            num_gqa_groups=self.num_kv_heads,
            input_layernorm=False,
            normalization=self.normalization,
            bias=use_bias,
            device=torch.cuda.current_device(),
            get_rng_state_tracker=get_cuda_rng_tracker,
            set_parallel_mode=self.set_parallel_mode,
            sequence_parallel=self.sequence_parallel,
            tp_group=self.tp_group,
            tp_size=self.world_size,
            params_dtype=self.params_dtype,
            return_bias=use_bias,
            qkv_format="sbhd",
            fuse_qkv_params=True,
        )

        self.rope_emb = None
        if neox_args.pos_emb == "rotary":
            self.hidden_size_per_attention_head = mpu.divide(
                neox_args.hidden_size, neox_args.num_attention_heads
            )

            if neox_args.rotary_pct == 1:
                self.rotary_ndims = None
            else:
                assert neox_args.rotary_pct < 1
                self.rotary_ndims = int(
                    self.hidden_size_per_attention_head * neox_args.rotary_pct
                )
            dim = (
                self.rotary_ndims
                if self.rotary_ndims is not None
                else self.hidden_size_per_attention_head
            )
            self.rotary_embeddings = RotaryEmbedding(
                dim,
                base=neox_args.rotary_emb_base,
                max_seq_len=neox_args.seq_length,
                precision=neox_args.params_dtype,
                save_inv_freqs=neox_args.rotary_save_freqs_buffer,
            )
            self.rope_emb = self.rotary_embeddings.get_emb()

    def forward(
        self,
        hidden_states,
        attention_mask,
        layer_past=None,
        rope_emb=None,
        *,
        cu_seqlens=None,
        num_documents=None,
        max_seqlen=None,
        **kwargs,
    ):
        if cu_seqlens is None:
            result = super(TEMultiheadAttention, self).forward(
                hidden_states, attention_mask, rotary_pos_emb=self.rope_emb, **kwargs
            )
        else:
            cu_seqlens = cu_seqlens[: int(num_documents.item()) + 1]
            max_seqlen = int(max_seqlen.item())
            self.qkv_format = "thd"
            try:
                result = super(TEMultiheadAttention, self).forward(
                    hidden_states.squeeze(1),  # Packed TE input is [tokens, hidden].
                    None,
                    rotary_pos_emb=self.rope_emb,
                    attn_mask_type="padding_causal",
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_kv=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_kv=max_seqlen,
                    **kwargs,
                )
            finally:
                self.qkv_format = "sbhd"

        output, bias = result if self.return_bias else (result, None)
        if cu_seqlens is not None:
            # THD output is [T, H]; restore the singleton batch dimension.
            output = output.unsqueeze(1)
        return output, bias


class TEDelayedScaling:
    """Configure the TE delayed-scaling recipe used for a NeoX block."""

    def __init__(self, neox_args):
        if neox_args.te_fp8_format == "e4m3":
            fp8_format = te.common.recipe.Format.E4M3
        elif neox_args.te_fp8_format == "hybrid":
            fp8_format = te.common.recipe.Format.HYBRID
        else:
            raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

        precision_kwargs = {}
        if not neox_args.te_fp8_wgrad:
            if not hasattr(
                te.common.recipe.DelayedScaling, "override_linear_precision"
            ):
                raise ValueError(
                    "te_fp8_wgrad=False requires Transformer Engine support for "
                    "high-precision weight gradients with delayed scaling. "
                    "Set te_fp8_wgrad=True or use the TE 1.12 image."
                )
            precision_kwargs["override_linear_precision"] = (False, False, True)

        self.recipe = te.common.recipe.DelayedScaling(
            margin=neox_args.te_fp8_margin,
            fp8_format=fp8_format,
            amax_compute_algo=neox_args.te_fp8_amax_compute_algo,
            amax_history_len=neox_args.te_fp8_amax_history_len,
            fp8_dpa=neox_args.te_fp8_mha,
            fp8_mha=neox_args.te_fp8_mha,
            **precision_kwargs,
        )

    def get_context(self):
        # Keep process groups out of the recipe serialized by TE checkpoints.
        group = get_tensor_model_parallel_group()
        if hasattr(te.pytorch, "autocast"):
            return te.pytorch.autocast(
                enabled=True, recipe=self.recipe, amax_reduction_group=group
            )
        return te.pytorch.fp8_autocast(
            enabled=True, fp8_recipe=self.recipe, fp8_group=group
        )

    def register_checkpoint_hook(self, layer):
        if Version(te.__version__) < Version("2.21"):
            return
        if hasattr(layer, "register_load_state_dict_pre_hook"):
            layer.register_load_state_dict_pre_hook(self._restore_legacy_recipe)
        else:  # Older PyTorch exposes the same hook through its private API.
            layer._register_load_state_dict_pre_hook(
                self._restore_legacy_recipe, with_module=True
            )

    def _restore_legacy_recipe(self, layer, state_dict, prefix, *_):
        # Leave TE's default handling of untrusted extra state in effect.
        if os.getenv("NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE") != "1":
            return
        from transformer_engine.pytorch.module.base import TransformerEngineBaseModule

        for name, module in layer.named_modules():
            if not isinstance(module, TransformerEngineBaseModule):
                continue
            key = f"{prefix}{name}._extra_state"
            buffer = state_dict.get(key)
            if not isinstance(buffer, BytesIO):
                continue
            state = torch.load(
                BytesIO(buffer.getbuffer()), map_location="cpu", weights_only=True
            )
            if state is None or "recipe" in state:
                continue
            # 1.12 saved the scaling tensors without a recipe. TE 2.21 would
            # allocate a default 1024-row window before copying these tensors.
            length = state["amax_history_fwd"].size(0)
            if length != state["amax_history_bwd"].size(0):
                raise ValueError(f"Inconsistent FP8 amax history lengths in {key}")
            state["recipe"] = replace(self.recipe, amax_history_len=length)
            restored = BytesIO()
            torch.save(state, restored)
            restored.seek(0)
            state_dict[key] = restored

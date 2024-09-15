import math

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from megatron.mpu.initialize import get_model_parallel_rank
from megatron.mpu.initialize import get_model_parallel_world_size
from megatron.mpu.initialize import get_tensor_model_parallel_group
from megatron.mpu.mappings import copy_to_model_parallel_region
from megatron.mpu.mappings import gather_from_model_parallel_region
from megatron.mpu.mappings import reduce_from_model_parallel_region
from megatron.mpu.mappings import scatter_to_model_parallel_region
from megatron.mpu.mappings import reduce_scatter_to_sequence_parallel_region
from megatron.mpu.mappings import gather_from_sequence_parallel_region
from megatron.mpu.random import get_cuda_rng_tracker
from megatron.mpu.utils import divide
from megatron.mpu.utils import VocabUtility
from functools import partial

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
    def __init__(self, in_features, out_features, bias=True):

        super(TELinear, self).__init__(in_features,out_features,bias)
        

    #     self.linear = te.pytorch.Linear(in_features, out_features, bias=use_bias, init_method=weight, **kwargs)


    # def forward(self, x):
    #     return self.linear(x)


class TELayerNormColumnParallelLinear(te.pytorch.LayerNormLinear):
    """
    Wrapper for the Transformer-Engine's `LayerNormLinear` layer that combines
    layernorm and linear layers
    """

    def __init__(self):
        # TODO
        return

    def forward(self, x):
        # TODO
        return


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
        mup_rescale_parameters=False,
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
        self.mup_rescale_parameters = mup_rescale_parameters
        self.use_mup = neox_args.use_mup
        self.params_dtype=neox_args.params_dtype
        self.parallel_mode="column"
        # print("##########################")
        # print(self.return_bias)

        super(TEColumnParallelLinear, self).__init__(in_features=self.input_size, out_features=self.output_size,
        bias= self.use_bias, init_method=self.init_method, get_rng_state_tracker=get_cuda_rng_tracker,
        device=torch.cuda.current_device(), sequence_parallel=self.sequence_parallel, tp_group=self.tp_group,
        tp_size=self.world_size, parallel_mode=self.parallel_mode, return_bias=self.skip_bias_add,
        params_dtype=self.params_dtype)

    # Copied from Mup
    def width_mult(self):
        assert hasattr(self.weight, "infshape"), (
            "Please call set_base_shapes(...). If using torch.nn.DataParallel, "
            "switch to distributed training with "
            "torch.nn.parallel.DistributedDataParallel instead"
        )
        return self.weight.infshape.width_mult()

    def set_parallel_output(self, value: bool):
        assert isinstance(value, bool)
        self.gather_output = (
            not value
        )  # if gather_output is True, parallel output is False, so we set the opposite

    # Copied from Mup
    def _rescale_parameters(self):
        """Rescale parameters to convert SP initialization to μP initialization.
        Warning: This method is NOT idempotent and should be called only once
        unless you know what you are doing.
        """
        if hasattr(self, "_has_rescaled_params") and self._has_rescaled_params:
            raise RuntimeError(
                "`_rescale_parameters` has been called once before already. "
                "Unless you know what you are doing, usually you should not be calling `_rescale_parameters` more than once.\n"
                "If you called `set_base_shapes` on a model loaded from a checkpoint, "
                "or just want to re-set the base shapes of an existing model, "
                "make sure to set the flag `rescale_params=False`.\n"
                "To bypass this error and *still rescale parameters*, set `self._has_rescaled_params=False` before this call."
            )
        if self.bias is not None:
            self.bias.data *= self.width_mult() ** 0.5
        self.weight.data *= self.width_mult() ** 0.5
        self._has_rescaled_params = True

    def mup_reinitialize_weights(self, neox_args):
        if neox_args.use_cpu_initialization:
            self.master_weight = _initialize_affine_weight_cpu(
                neox_args,
                self.weight,
                self.output_size,
                self.input_size,
                self.output_size_per_partition,
                0,
                partial(self.init_method, use_mup=True),
                stride=self.stride,
                return_master_weight=keep_master_weight_for_test,
            )
        else:
            _initialize_affine_weight_gpu(
                self.weight,
                partial(self.init_method, use_mup=True),
                partition_dim=0,
                stride=self.stride,
            )
    
    def forward(self, inp, **kwargs):
        if self.use_mup and self.mup_rescale_parameters:
            input_ /= self.width_mult()
        
        output = super(TEColumnParallelLinear, self).forward(inp, **kwargs)

        if self.gather_output:
            # All-gather across the partitions.
            assert (
                not self.sequence_parallel
            ), "sequence_parallel=True and gather_output=True are incompatible!"
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel

        if self.skip_bias_add:
            return output
        else:
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
        mup_rescale_parameters=False,
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
        self.mup_rescale_parameters = mup_rescale_parameters
        self.use_mup = neox_args.use_mup
        self.params_dtype=neox_args.params_dtype
        self.parallel_mode="row"
        
        # if self.input_is_parallel:
        #     self.input_size = divide(self.input_size, self.world_size)

        super(TERowParallelLinear, self).__init__(in_features=self.input_size, out_features=self.output_size,
        bias= self.use_bias, init_method=self.init_method, get_rng_state_tracker=get_cuda_rng_tracker,
        device=torch.cuda.current_device(), sequence_parallel=self.sequence_parallel, tp_group=self.tp_group,
        tp_size=self.world_size, parallel_mode=self.parallel_mode, return_bias=self.skip_bias_add,
        params_dtype=self.params_dtype)

    # Copied from Mup
    def width_mult(self):
        assert hasattr(self.weight, "infshape"), (
            "Please call set_base_shapes(...). If using torch.nn.DataParallel, "
            "switch to distributed training with "
            "torch.nn.parallel.DistributedDataParallel instead"
        )
        return self.weight.infshape.width_mult()

    # Copied from Mup
    def _rescale_parameters(self):
        """Rescale parameters to convert SP initialization to μP initialization.
        Warning: This method is NOT idempotent and should be called only once
        unless you know what you are doing.
        """
        if hasattr(self, "_has_rescaled_params") and self._has_rescaled_params:
            raise RuntimeError(
                "`_rescale_parameters` has been called once before already. "
                "Unless you know what you are doing, usually you should not be calling `_rescale_parameters` more than once.\n"
                "If you called `set_base_shapes` on a model loaded from a checkpoint, "
                "or just want to re-set the base shapes of an existing model, "
                "make sure to set the flag `rescale_params=False`.\n"
                "To bypass this error and *still rescale parameters*, set `self._has_rescaled_params=False` before this call."
            )
        if self.bias is not None:
            self.bias.data *= self.width_mult() ** 0.5
        self.weight.data *= self.width_mult() ** 0.5
        self._has_rescaled_params = True

    def mup_reinitialize_weights(self, neox_args):
        if neox_args.use_cpu_initialization:
            self.master_weight = _initialize_affine_weight_cpu(
                neox_args,
                self.weight,
                self.output_size,
                self.input_size,
                self.input_size_per_partition,
                1,
                partial(self.init_method, use_mup=True),
                stride=self.stride,
                return_master_weight=self.keep_master_weight_for_test,
            )
        else:
            _initialize_affine_weight_gpu(
                self.weight,
                partial(self.init_method, use_mup=True),
                partition_dim=1,
                stride=self.stride,
            )

    def set_parallel_output(self, parallel_output: bool):
        assert isinstance(parallel_output, bool)
        self.parallel_output = parallel_output
    
    def forward(self, inp, **kwargs):
        # if not self.input_is_parallel:
        #     inp = scatter_to_model_parallel_region(inp)
        
        output = super(TERowParallelLinear, self).forward(inp, **kwargs)
        if self.skip_bias_add:
            return output
        else:
            return output, None


class TEDotProductAttention(te.pytorch.DotProductAttention):
    """
    Wrapper for the Transformer-Engine's `DotProductAttention` layer that also
    has "flash attention" enabled.
    """

    def __init__(self):
        # TODO
        return

    def forward(self, x):
        # TODO
        return


class TEDelayedScaling(te.common.recipe.DelayedScaling):
    """
    Wrapper for the Transformer-Engine's `DelayedScaling` layer.
    """

    def __init__(self):
        # TODO
        return

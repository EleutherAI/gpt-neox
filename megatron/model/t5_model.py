# coding=utf-8
#
# Copyright 2022 EleutherAI Contributors. This file is based on code by the authors denoted below and has been modified from its original version.
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

"""T5 model."""

from collections import defaultdict
import math
from megatron.model.utils import SequentialWrapper, recursive_setattr
import torch
import torch.nn as nn

from functools import partial
from typing import Union, List

from megatron.model.norms import get_norm
from megatron.model.init_functions import get_init_methods

from megatron import mpu
from megatron.mpu import ParallelRelativePositionBias
from megatron.model.transformer import (
    ParallelTransformerLayerPipe,
    NormPipe,
    ParallelLinearPipe,
    ParallelEncoderDecoderLinearPipe,
    parallel_lm_logits,
    ParallelLinear,
)
from megatron.model.word_embeddings import (
    EncoderEmbeddingPipe, 
    DecoderEmbeddingPipe,
    EmbeddingPipe,
)

# Pipeline parallelism
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec

# TODO(Hailey): confirm crossentropy behavior is same as decoder-only
def gpt2_attention_mask_func(attention_scores, ltor_mask):
    attention_scores.masked_fill_(ltor_mask, -10000.0)
    return attention_scores


def cross_entropy(output, labels, _fp16=False):
    """From pretrain_gpt2:forward_step()"""
    """
    if self.fp16_lm_cross_entropy:
        assert output.dtype == torch.half
        loss = mpu.vocab_parallel_cross_entropy(output, labels)
    else:
        loss = mpu.vocab_parallel_cross_entropy(output.float(), labels)
        return loss
    """
    labels, loss_mask = labels[0], labels[1]
    if _fp16:
        assert output.dtype == torch.half and loss_mask.dtype == torch.half
        losses = mpu.vocab_parallel_cross_entropy(output.contiguous(), labels)
    else:
        losses = mpu.vocab_parallel_cross_entropy(output.float().contiguous(), labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss


def _pre_encoder_block(args):
    # data format change for hidden_states to avoid explicit tranposes : [b s h] --> [s b h]
    assert len(args) == 5, "Incorrect number of arguments to _pre_encoder_block"
    fn = lambda _args: (_args[0].transpose(0, 1).contiguous(), *_args[1:])
    return fn(args)


def _pre_decoder_block(args):
    # reformat inputs before passing them to decoder stack.
    assert len(args) == 4, "Incorrect number of arguments to _pre_decoder_block"
    fn = lambda _args: (_args[0].transpose(0, 1).contiguous(), *args[1:])
    return fn(args)


def _post_decoder_block(args):
    # drop unneeded vars and keep only decoder hidden states.
    # from (hidden_states, encoder_hidden_states, encoder_attention_mask, attention_mask)
    # to (hidden_states.T)
    assert len(args) == 4, "Incorrect number of arguments to _post_decoder_block"
    fn = lambda _args: (_args[0].transpose(0, 1).contiguous())
    return fn(args)


def _post_encoder_decoder_block(args):
    # data format change for hidden_states to avoid explicit tranposes : [b s h] --> [s b h]
    assert len(args) == 4, "Incorrect number of arguments to _post_encoder_block"
    fn = lambda _args: (
        _args[0].transpose(0, 1).contiguous(),
        _args[1].transpose(0, 1).contiguous()
        )
    return fn(args)


class T5ModelPipe(PipelineModule, torch.nn.Module):
    """T5Model adapted for pipeline parallelism.
    :param neox_args
    :param num_tokentypes: number of token types (TODO): deprecated?
    :param parallel_output: if true, don't gather the output logits, and calculate loss in parallel. Set to true by default in training for efficiency, but set to false for inference.
    :param topology: deepspeed topology object specifying pipe / model parallelism topology.
    :param use_cache: if true, cache key/value pairs for each layer in inference.
    """

    def __init__(
        self,
        neox_args,
        num_tokentypes=0,
        parallel_output=True,
        topology=None,
        use_cache=False,
    ):
        self.neox_args = neox_args

        self.use_cache = use_cache
        self.parallel_output = parallel_output
        self.hidden_size = self.neox_args.hidden_size #TODO(Hailey): pass in any other sizes/num layers?
        self.num_tokentypes = num_tokentypes
        self.init_method, self.output_layer_init_method = get_init_methods(
            self.neox_args
        )

        self.__topology__ = topology

        self.specs = []
        self.init_specs()

        self.checkpointable_layers = ["ParallelTransformerLayerPipe"]

        super().__init__(
            layers=self.specs,
            loss_fn=partial(cross_entropy, _fp16=self.neox_args.fp16_lm_cross_entropy),
            topology=topology,
            activation_checkpoint_interval=self.neox_args.checkpoint_num_layers
            if self.neox_args.checkpoint_activations 
            else 0,
            partition_method=neox_args.pipe_partition_method,
            checkpointable_layers=self.checkpointable_layers,
        )

    def insert_layers(
        self, layers: Union[nn.Module, nn.ModuleList, nn.Sequential, List], idx
    ):
        """
        inserts the layers in `layers` into the pipe model at `idx`.
        """
        if isinstance(layers, nn.Module):
            self.specs.insert(idx, layers)
        elif any(
            [isinstance(layers, nn.ModuleList), isinstance(layers, nn.Sequential)]
        ):
            self.specs[idx:idx] = layers
        elif isinstance(layers, list):
            assert all(
                [hasattr(l, "__call__") for l in layers]
            ), "all items in `layers` must be Callables"
            self.specs[idx:idx] = layers
        else:
            raise ValueError(
                f"layer passed into {self.__class__.__name__}.insert_layer() should be either an nn.Module, an nn.ModuleList, an nn.Sequential object, or a list of callables not a {type(layers)}"
            )

        # re-initialize parent class
        super().__init__(
            layers=self.specs,
            loss_fn=self.loss_fn,
            topology=self.__topology__,
            activation_checkpoint_interval=self.activation_checkpoint_interval,
            partition_method=self.neox_args.pipe_partition_method,
            checkpointable_layers=self.checkpointable_layers,
        )

    def init_specs(self):

        weight_tying = not self.neox_args.no_weight_tying
        self.specs = []

        # embedding layer
        # input format we want: 
        # (encoder_input_ids, decoder_input_ids, encoder_position_ids, decoder_position_ids, encoder_attn_mask, (decoder)attention_mask)

        if weight_tying:
            self.specs.append(
                TiedLayerSpec(
                    "embed",
                    EncoderEmbeddingPipe,
                    self.neox_args,
                    self.hidden_size,
                    self.neox_args.padded_vocab_size,
                    self.neox_args.max_position_embeddings,
                    self.neox_args.hidden_dropout,
                    self.init_method,
                    self.num_tokentypes,
                    tied_weight_attr="word_embeddings_weight",
                )
            )
        else:
            self.specs.append(
                LayerSpec(
                    EncoderEmbeddingPipe,
                    self.neox_args,
                    self.hidden_size,
                    self.neox_args.padded_vocab_size,
                    self.neox_args.max_position_embeddings,
                    self.neox_args.hidden_dropout,
                    self.init_method,
                    self.num_tokentypes,
                )
            )
        
        # per gpt2_model.py, attention mask MUST be the last item in args
        # passed from one stage to the next.

        # current output format: (hidden_states, decoder_input_ids, decoder_position_ids, enc attn mask, attention_mask)

        self.specs.append(_pre_encoder_block)
        
        # T5-style RPE positional embedding
        if self.neox_args.pos_emb == "rpe":
            hidden_size_per_attention_head = mpu.divide(
                self.hidden_size, self.neox_args.num_attention_heads
            )
            rpe_scale = math.sqrt(hidden_size_per_attention_head)
            rpe_emb = ParallelRelativePositionBias(
                neox_args=self.neox_args,
                scale=rpe_scale,
                causal=False, #TODO(Hailey): I think we want causal=false. but confirm this
                num_buckets=self.neox_args.rpe_num_buckets,
                max_distance=self.neox_args.rpe_max_distance,
                heads=self.neox_args.num_attention_heads,
            )

        # transformer encoder layers
        for i in range(self.neox_args.num_encoder_layers):
            layer_type = self.neox_args.attention_config[i]
            self.specs.append(
                LayerSpec(
                    ParallelTransformerLayerPipe,
                    neox_args=self.neox_args,
                    attention_mask_func=gpt2_attention_mask_func,
                    init_method=self.init_method,
                    output_layer_init_method=self.output_layer_init_method,
                    layer_number=i,
                    layer_type="encoder",
                    rpe=rpe_emb if self.neox_args.pos_emb == "rpe" else None,
                    rotary=self.neox_args.pos_emb == "rotary",
                    use_cache=self.use_cache,
                )
            )

        # drop attn masks and encoder hidden states, and reshape decoder hidden states
        # self.specs.append(_post_encoder_block)

        # output format now not a tuple, just: hidden_states

        # current output format: (hidden_states, decoder_input_ids, decoder_position_ids, enc attn mask, attention_mask)
        
        # decoder emb layer 
        if weight_tying:
            self.specs.append(
                TiedLayerSpec(
                    "embed",
                    DecoderEmbeddingPipe,
                    self.neox_args,
                    self.hidden_size,
                    self.neox_args.padded_vocab_size,
                    self.neox_args.max_position_embeddings,
                    self.neox_args.hidden_dropout,
                    self.init_method,
                    self.num_tokentypes,
                    tied_weight_attr="word_embeddings_weight",
                )
            )
        else:
            self.specs.append(
                LayerSpec(
                    DecoderEmbeddingPipe,
                    self.neox_args,
                    self.hidden_size,
                    self.neox_args.padded_vocab_size,
                    self.neox_args.max_position_embeddings,
                    self.neox_args.hidden_dropout,
                    self.init_method,
                    self.num_tokentypes,
                )
            )

        self.specs.append(_pre_decoder_block)
        # current output format:  (decoder_hidden_states, encoder_hidden_states, encoder_attention_mask, attention_mask)
        
        # transformer decoder layers # TODO(Hailey): right now, neox.num_layers = the number of decoder layers for minimal code change to rest of repo. update this later
        for i in range(self.neox_args.num_encoder_layers, self.neox_args.num_layers):
            layer_type = self.neox_args.attention_config[i]
            self.specs.append(
                LayerSpec(
                    ParallelTransformerLayerPipe, 
                    neox_args=self.neox_args,
                    attention_mask_func=gpt2_attention_mask_func,
                    init_method=self.init_method,
                    output_layer_init_method=self.output_layer_init_method,
                    layer_number=i,
                    layer_type="decoder",
                    rpe=rpe_emb if self.neox_args.pos_emb == "rpe" else None,
                    rotary=self.neox_args.pos_emb == "rotary",
                    use_cache=self.use_cache,
                )
            )
        
        # drop attn masks and encoder hidden states, and reshape decoder hidden states
        # self.specs.append(_post_decoder_block)
        self.specs.append(_post_encoder_decoder_block)

        # per gpt2.model.py NormPipe is deprecated...
        norm, eps = get_norm(self.neox_args)
        self.specs.append(
            LayerSpec(NormPipe, norm, self.neox_args.hidden_size, eps=eps)
        )

        def _logits_helper(embedding, lm_output):
            logits = parallel_lm_logits(
                lm_output, embedding.word_embeddings_weight, self.parallel_output
            )
            return logits

        if weight_tying:
            self.specs.append(
                TiedLayerSpec(
                    "embed",
                    EmbeddingPipe,
                    self.neox_args,
                    self.hidden_size,
                    self.neox_args.padded_vocab_size,
                    self.neox_args.max_position_embeddings,
                    self.neox_args.hidden_dropout,
                    self.init_method,
                    self.num_tokentypes,
                    forward_fn=_logits_helper,
                    tied_weight_attr="word_embeddings_weight",
                )
            )
        else:
            self.specs.append(
                LayerSpec(
                    # ParallelLinearPipe,
                    ParallelEncoderDecoderLinearPipe
                    neox_args=self.neox_args,
                    init_method=self.init_method,
                    parallel_output=self.parallel_output,
                )
            )
    
    def _set_parallel_output(self, value):
        # set the parallel output value for the final layer to value
        final_layer = list(self.forward_funchs)[-1]
        if isinstance(final_layer, (ParallelLinearPipe, ParallelLinear)):
            final_layer.final_linear.set_parallel_output(value)
    
    def inference_mode(self, use_cache=True):
        """
        Sets up the model for inference by turning on k/v caching (if specified) and setting `parallel output` of the final layer to false,
        so logits are gathered across model parallel ranks.
        :param cache: (bool) True if you want to use caching during inference, False otherwise
        """
        # set caching to true if specified
        recursive_setattr(self.forward_funcs, "use_cache", use_cache, assert_type=bool)
        # set parallel output to false so gathering is done automatically
        self._set_parallel_output(False)

    def train_mode(self):
        """
        Sets up the model for training by turning off k/v caching and setting `parallel output` of the final layer to true,
        so logits are gathered across model parallel ranks.
        """
        # set caching to false
        recursive_setattr(self.forward_funcs, "use_cache", False)
        # set parallel output to true (more efficient for training)
        self._set_parallel_output(True)

    def clear_cache(self):
        """
        Clears the kv cache of the model on all layers.
        """
        recursive_setattr(self.forward_funcs, "layer_past", None)

    def to_sequential(self):
        """
        Convert PipelineModule to nn.Sequential module.
        :return:
        """
        layers = []
        tied_layers = defaultdict(list)
        for n, spec in enumerate(self.specs):
            if isinstance(spec, TiedLayerSpec):
                if spec.key in tied_layers:
                    # receiver
                    layers.append(
                        Lambda(lambda x: spec.forward_fn(tied_layers[spec.key][0], x))
                    )
                else:
                    # owner
                    module = spec.build(log=False)
                    layers.append(module)
                    tied_layers[spec.key].append(module)
            elif isinstance(spec, LayerSpec):
                layers.append(spec.build(log=False))
            elif hasattr(spec, "__call__"): # check callable
                layers.append(Lambda(spec))
            else:
                raise ValueError(f"Layer number {n} ({spec}) not recognized")
        model = SequentialWrapper(
            layers,
            self.activation_checkpoint_interval,
            self.activation_checkpoint_func,
            parent_class_name=self.__class__.__name__,
        )
        return model

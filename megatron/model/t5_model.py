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
from mimetypes import init
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
    parallel_lm_logits,
    ParallelLinear,
)
from megatron.model.word_embeddings import EmbeddingPipe

# Pipeline parallelism
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec

# TODO(Hailey): remove this fn? look for if it's needed
def t5_extended_attention_mask(attention_mask_list):

    def attn_mask_postprocess(attn_mask):
        # [b, 1, s, s]
        extended_attention_mask = attn_mask.unsqueeze(1)
        return extended_attention_mask

    return [attn_mask_postprocess(attn_mask) for attn_mask in attention_mask_list]

# TODO(Hailey): remove this fn?
def t5_position_ids(token_ids):
    # Create position ids
    seq_length = token_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=token_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(token_ids)

    return position_ids

# TODO(Hailey): can these next 3 fns get imported from gpt2 model?
def cross_entropy(output, labels, _fp16=False):
    raise NotImplementedError

def _pre_transformer_block(args)
    raise NotImplementedError

def _post_transformer_block(args)
    raise NotImplementedError

class T5ModelPipe(PipelineModule, torch.nn.Module):
    """T5Model adapted for pipeline parallelism.
    Changes: TODO(Hailey)
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
        self.init_specs() # TODO(Hailey): what are specs used for?

        self.checkpointable_layers = [] #TODO(Hailey): add checkpointable layers here

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
        # input format: (input_ids, position_ids, attention_mask)

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
                    tied_weight_attr="word_embeddings_weight",
                )
            )
        else:
            self.specs.append(
                LayerSpec(
                    EmbeddingPipe,
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

        # output format: (hidden_states, attention_mask)

        self.specs.append(_pre_transformer_block) # TODO(Hailey): make sure these fns are still needed, move them from gpt2_model.py to some utils file if so?

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
                    # TODO(Hailey): decide whether this requires a different EncoderLayer class
                    neox_args=self.neox_args,
                    attention_mask_func=gpt2_attention_mask_func, #TODO(Hailey): add this fn to this file
                    init_method=self.init_method,
                    output_layer_init_method=self.output_layer_init_method,
                    layer_number=i,
                    rpe=rpe_emb if self.neox_args.pos_emb == "rpe" else None,
                    rotary=self.neox_args.pos_emb == "rotary",
                    use_cache=self.use_cache,
                )
            )
        
        # transformer decoder layers # TODO(Hailey): right now, num_layers = the number of decoder layers for minimal code change to rest of repo. update this later
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
                    rpe=rpe_emb if self.neox_args.pos_emb == "rpe" else None,
                    rotary=self.neox_args.pos_emb == "rotary",
                    use_cache=self.use_cache,
                )
            )
        
        # drop attn mask and reshape hidden states
        self.specs.append(_post_transformer_block)

        # per gpt2.model.py NormPipe is deprecated...
        norm, eps = get_norm(self.neox_args)
        self.specs.append(
            LayerSpec(NormPipe, norm, self.neox_args.hidden_size, eps=eps)
        )

        # output format now not a tuple, just: hidden_states

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
                    ParallelLinearPipe,
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

    

# class T5LMHead(MegatronModule):
#     """Masked LM head for T5
#     Arguments:
#         mpu_vocab_size: model parallel size of vocabulary.
#         hidden_size: hidden size
#         init_method: init method for weight initialization
#         layernorm_epsilon: tolerance for layer norm divisions
#         parallel_output: wether output logits being distributed or not.
#     """

#     def __init__(self, mpu_vocab_size, parallel_output):
#         super(T5LMHead, self).__init__()

#         args = get_args()

#         self.bias = torch.nn.Parameter(torch.zeros(mpu_vocab_size))
#         self.bias.model_parallel = True
#         self.bias.partition_dim = 0
#         self.bias.stride = 1
#         self.parallel_output = parallel_output

#     def forward(self, hidden_states, word_embeddings_weight):
#         output = parallel_lm_logits(hidden_states,
#                                     word_embeddings_weight,
#                                     self.parallel_output,
#                                     bias=self.bias)
#         return output


# class T5Model(MegatronModule):
#     """T5 Language model."""

#     def __init__(self, num_tokentypes=0, parallel_output=True):
#         super(T5Model, self).__init__()
#         args = get_args()

#         self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
#         self.parallel_output = parallel_output
#         init_method = init_method_normal(args.init_method_std)
#         scaled_init_method = scaled_init_method_normal(args.init_method_std,
#                                                        args.num_layers)

#         self.language_model, self._language_model_key = get_language_model(
#             num_tokentypes=num_tokentypes,
#             add_pooler=False,
#             add_decoder=True,
#             encoder_attn_mask_type=AttnMaskType.padding,
#             init_method=init_method,
#             scaled_init_method=scaled_init_method)

#         self.lm_head = T5LMHead(
#             self.language_model.embedding.word_embeddings.weight.size(0),
#             parallel_output)
#         self._lm_head_key = 'lm_head'

#     def set_input_tensor(self, input_tensor):
#         """See megatron.model.transformer.set_input_tensor()"""
#         self.language_model.set_input_tensor(input_tensor)

#     def forward(self, encoder_input_ids, decoder_input_ids, encoder_attn_mask,
#                 decoder_attn_mask, encoder_decoder_attn_mask,
#                 tokentype_ids=None, lm_labels=None, enc_hidden_states=None):

#         # Converting the attention masks to proper parameter settings
#         encoder_attn_mask, decoder_attn_mask, encoder_decoder_attn_mask = t5_extended_attention_mask(
#             [encoder_attn_mask, decoder_attn_mask, encoder_decoder_attn_mask])

#         encoder_position_ids = t5_position_ids(encoder_input_ids)
#         decoder_position_ids = t5_position_ids(decoder_input_ids)

#         lm_output = self.language_model(encoder_input_ids,
#                                         encoder_position_ids,
#                                         encoder_attn_mask,
#                                         decoder_input_ids,
#                                         decoder_position_ids,
#                                         decoder_attn_mask,
#                                         encoder_decoder_attn_mask,
#                                         tokentype_ids=tokentype_ids,
#                                         enc_hidden_states=enc_hidden_states)

#         decoder_output, encoder_output = lm_output

#         # Output.
#         lm_logits = self.lm_head(decoder_output,
#                                  self.language_model.embedding.word_embeddings.weight)

#         if lm_labels is None:
#             return lm_logits, encoder_output
#         else:
#             if self.fp16_lm_cross_entropy:
#                 assert lm_logits.dtype == torch.half
#                 lm_loss = mpu.vocab_parallel_cross_entropy(lm_logits, lm_labels)
#             else:
#                 lm_loss = mpu.vocab_parallel_cross_entropy(lm_logits.float(),
#                                                            lm_labels)
#             return lm_loss, encoder_output

#     def state_dict_for_save_checkpoint(self, destination=None, prefix='',
#                                        keep_vars=False):
#         """For easy load when model is combined with other heads,
#         add an extra key."""

#         state_dict_ = {}
#         state_dict_[self._language_model_key] \
#             = self.language_model.state_dict_for_save_checkpoint(
#             destination, prefix, keep_vars)
#         state_dict_[self._lm_head_key] \
#             = self.lm_head.state_dict_for_save_checkpoint(
#             destination, prefix, keep_vars)
#         return state_dict_

#     def load_state_dict(self, state_dict, strict=True):
#         """Customized load."""

#         self.language_model.load_state_dict(
#             state_dict[self._language_model_key], strict=strict)
#         self.lm_head.load_state_dict(state_dict[self._lm_head_key],
#                                      strict=strict)

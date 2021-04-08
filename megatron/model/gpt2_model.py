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

"""GPT-2 model."""

import torch

from megatron import get_args
from megatron.module import MegatronModule
from functools import partial
from .language_model import get_language_model
from .utils import init_method_normal
from .utils import scaled_init_method_normal
from .norms import LayerNorm, RMSNorm, ScaleNorm

# Pipeline parallelism
from megatron import mpu
from megatron.mpu import ParallelRelativePositionBias
import megatron.fp16 as fp16
from megatron.model.transformer import ParallelTransformerLayerPipe, NormPipe, RowParallelLinearPipe
from .language_model import EmbeddingPipe, parallel_lm_logits
from megatron import print_rank_0

from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec


def gpt2_attention_mask_func(attention_scores, ltor_mask):
    attention_scores.masked_fill_(ltor_mask, -10000.0)
    return attention_scores


def cross_entropy(output, labels, _fp16=False):
    """ From pretrain_gpt2:forward_step() """
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
        assert (output.dtype == torch.half and loss_mask.dtype == torch.half)
        losses = mpu.vocab_parallel_cross_entropy(output.contiguous(), labels)
    else:
        output = fp16.fp16_to_fp32(output)
        losses = mpu.vocab_parallel_cross_entropy(output.contiguous(), labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss


class GPT2Model(MegatronModule):
    """GPT-2 Language model."""

    def __init__(self, num_tokentypes=0, parallel_output=True, inference=False, get_key_value=True):
        super(GPT2Model, self).__init__()
        args = get_args()
        self.parallel_output = parallel_output
        self.weight_tying = not args.no_weight_tying
        if not self.weight_tying:
            # TODO: not sure whether to use RowParallelLinear's default scatter to mp region here, or copy, which is
            # the default of parallel_lm_logits. Should investigate benefits of both
            self.final_linear = mpu.RowParallelLinear(
                args.hidden_size,
                args.padded_vocab_size,
                bias=False,
                input_is_parallel=False,
                skip_bias_add=False,
                parallel_output=self.parallel_output)

        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy

        self.inference = inference
        self.get_key_value = get_key_value if inference else False

        self.language_model, self._language_model_key = get_language_model(
            attention_mask_func=gpt2_attention_mask_func,
            num_tokentypes=num_tokentypes,
            init_method=init_method_normal(args.init_method_std),
            scaled_init_method=scaled_init_method_normal(args.init_method_std,
                                                         args.num_layers),
            get_key_value=self.get_key_value)


    def forward(self, input_ids, position_ids, attention_mask, 
                layer_past=None, tokentype_ids=None, forward_method_parallel_output=None, labels=None):

        # Language model.
        lm_output = self.language_model(input_ids,
                                        position_ids,
                                        attention_mask,
                                        tokentype_ids=tokentype_ids,
                                        layer_past=layer_past)

        if self.get_key_value:
            lm_output, presents = lm_output

        # Output.
        parallel_output = self.parallel_output
        if forward_method_parallel_output is not None:
            parallel_output = forward_method_parallel_output
        if self.weight_tying:
            output = parallel_lm_logits(
                lm_output,
                self.language_model.embedding.word_embeddings.weight,
                parallel_output)
        else:
            output, bias = self.final_linear(lm_output)

        if self.get_key_value:
            output = [output, presents]

        if labels is None:
            return output
        else:
            if self.fp16_lm_cross_entropy:
                assert output.dtype == torch.half
                loss = mpu.vocab_parallel_cross_entropy(output, labels)
            else:
                loss = mpu.vocab_parallel_cross_entropy(output.float(), labels)
            return loss

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        self.language_model.load_state_dict(state_dict, strict=strict)


class GPT2ModelPipe(PipelineModule, MegatronModule):
    """GPT2Model adapted for pipeline parallelism.

    The largest change is flattening the GPTModel class so we can express it as a
    sequence of layers including embedding, transformer layers, and output.
    """

    def __init__(self, num_tokentypes=0, parallel_output=True, topology=None, inference=False, get_key_value=True):
        args = get_args()

        self._inference = inference
        self.get_key_value = get_key_value if inference else False
        self.parallel_output = parallel_output
        self.hidden_size = args.hidden_size
        self.num_tokentypes = num_tokentypes
        self.init_method = init_method_normal(args.init_method_std)
        self.output_layer_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy

        #
        # forward() prototype
        # 
        self.specs = []
        self.init_specs(args)
        loss_fn = partial(cross_entropy, _fp16=self.fp16_lm_cross_entropy)
        if args.checkpoint_activations:
            interval = args.checkpoint_num_layers
        else:
            interval = 0
        super().__init__(layers=self.specs,
                         loss_fn=loss_fn if not self._inference else None,
                         topology=topology,
                         activation_checkpoint_interval=interval,
                         partition_method='type:transformer')

    def init_specs(self, args):
        weight_tying = not args.no_weight_tying
        if args.pos_emb == 'rpe':
            rpe_emb = ParallelRelativePositionBias(causal=True, num_buckets=args.rpe_num_buckets,
                                                   max_distance=args.rpe_max_distance,
                                                   heads=args.num_attention_heads)
        else:
            rpe_emb = None
        # Embedding layer
        # input will be (input_ids, position_ids, attention_mask) in Training
        # and (input_ids, position_ids, attention_mask, layer_past) in Inference
        if weight_tying:
            self.specs.append(TiedLayerSpec('embed',
                                            EmbeddingPipe,
                                            self.hidden_size,
                                            args.padded_vocab_size,
                                            args.max_position_embeddings,
                                            args.hidden_dropout,
                                            self.init_method,
                                            self.num_tokentypes,
                                            tied_weight_attr='word_embeddings_weight'))
        else:
            self.specs.append(LayerSpec(EmbeddingPipe,
                                        self.hidden_size,
                                        args.padded_vocab_size,
                                        args.max_position_embeddings,
                                        args.hidden_dropout,
                                        self.init_method,
                                        self.num_tokentypes))

        # NB: in inference, the attention mask always needs to be the *last* item in the args when being passed from 
        # one stage to the next, because deepspeed is hacks on top of hacks.
        #
        # outputs are now
        #           Train: (hidden_states, attention_mask)
        #           Inference: (hidden_states, layer_past, attention_mask)
        # 
        # data format change for hidden_states to avoid explicit tranposes : [b s h] --> [s b h]

        if self._inference:
            # we need to add a container to cache `presents` from each layer's forward pass
            # inputs/outputs are now (hidden_states, layer_past, presents, attention_mask)
            self.specs.append(lambda x: (x[0].transpose(0, 1).contiguous(), x[1], torch.Tensor(), x[2]))
        else:
            self.specs.append(lambda x: (x[0].transpose(0, 1).contiguous(), *x[1:]))

        # Transformer layers
        for x in range(args.num_layers):
            if args.sparsity == 'none':
                sparse = False
            elif args.sparsity == 'all':
                sparse = True
            elif args.sparsity == 'interspersed':
                sparse = not x % 2 == 0
            self.specs.append(
                LayerSpec(ParallelTransformerLayerPipe,
                          attention_mask_func=gpt2_attention_mask_func,
                          init_method=self.init_method,
                          output_layer_init_method=self.output_layer_init_method,
                          layer_number=x,
                          sparse=sparse,
                          rpe=rpe_emb,
                          get_key_value=self.get_key_value))
                          
        if self._inference:
            # we can get rid of the mask / pasts now
            # from (hidden_states, layer_past, presents, attention_mask)
            # to (hidden_states^T, presents)
            self.specs.append(lambda x: (x[0].transpose(0, 1).contiguous(), x[2]))
        else:
            # Undo data format change and drop mask
            self.specs.append(lambda x: x[0].transpose(0, 1).contiguous())

        # Final layernorm after transformer layers
        if args.norm == "rmsnorm":
            norm = RMSNorm
            eps = args.rms_norm_epsilon
        elif args.norm == "layernorm":
            eps = args.layernorm_epsilon
            norm = LayerNorm
        elif args.norm == "scalenorm":
            eps = args.scalenorm_epsilon
            norm = ScaleNorm

        # NormPipe is a helper class to pass presents through to the output when doing inference
        self.specs.append(
            LayerSpec(NormPipe,
                      norm,
                      args.hidden_size,
                      eps=eps))

        # outputs are now
        #           Train: hidden_states
        #           Inference: (hidden_states, presents)

        # XXX forward_method_parallel_output is assumed to be None, but we're not in a
        # fwd method to assert

        def _logits_helper(embedding, lm_output):
            """Just a wrapper to massage inputs/outputs from pipeline. """
            if self._inference and len(lm_output) == 2:
                hidden_states, presents = lm_output
                output = parallel_lm_logits(
                    hidden_states,
                    embedding.word_embeddings_weight,
                    self.parallel_output)
                return hidden_states, presents
            else:
                return parallel_lm_logits(
                    lm_output,
                    embedding.word_embeddings_weight,
                    self.parallel_output)

        if weight_tying:
            self.specs.append(
                TiedLayerSpec('embed',
                              EmbeddingPipe,
                              self.hidden_size,
                              args.padded_vocab_size,
                              args.max_position_embeddings,
                              args.hidden_dropout,
                              self.init_method,
                              self.num_tokentypes,
                              forward_fn=_logits_helper,
                              tied_weight_attr='word_embeddings_weight')
            )
        else:
            # TODO: not sure whether to use RowParallelLinear's default scatter to mp region here, or copy, which is
            # the default of parallel_lm_logits. Should investigate benefits of both
            self.specs.append(
                LayerSpec(
                    RowParallelLinearPipe,
                    args.hidden_size,
                    args.padded_vocab_size,
                    bias=False,
                    input_is_parallel=False,
                    parallel_output=self.parallel_output,
                    skip_bias_add=False
                )
            )
        # so output in training should just be logits
        # in inference it will be (logits, presents) (assuming get_key_value) is true


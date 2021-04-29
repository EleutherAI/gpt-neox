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
from megatron.model.utils import init_method_normal, scaled_init_method_normal, Lambda
from megatron.model.norms import LayerNorm, RMSNorm, ScaleNorm

from megatron import mpu
from megatron.mpu import ParallelRelativePositionBias
import megatron.fp16 as fp16
from megatron.model.transformer import ParallelTransformerLayerPipe, NormPipe, ParallelLinearPipe, parallel_lm_logits
from megatron.model.word_embeddings import EmbeddingPipe

# Pipeline parallelism
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
        self.embedding_type = args.pos_emb

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
        if self.embedding_type == 'rpe':
            rpe_emb = ParallelRelativePositionBias(causal=True, num_buckets=args.rpe_num_buckets,
                                                   max_distance=args.rpe_max_distance,
                                                   heads=args.num_attention_heads)
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy

        #
        # forward() prototype
        # 
        self.specs = []
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
        #           Train: (hidden_states, ((maybe) rotary_pos_emb), attention_mask)
        #           Inference: (hidden_states, layer_past, ((maybe) rotary_pos_emb), attention_mask)
        # 
        # data format change for hidden_states to avoid explicit tranposes : [b s h] --> [s b h]

        if self._inference:
            # we need to add a container to cache `presents` from each layer's forward pass
            # inputs/outputs are now (hidden_states, layer_past, presents, attention_mask)
            self.specs.append(lambda x: (x[0].transpose(0, 1).contiguous(), x[1], torch.Tensor(), *x[2:]))
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
                          rpe=rpe_emb if args.pos_emb == 'rpe' else None,
                          get_key_value=self.get_key_value,
                          rotary=args.pos_emb == 'rotary'))

        if self._inference:
            # we can get rid of the mask / pasts / (?rotary_pos_emb) now
            # from (hidden_states, layer_past, presents, (maybe rotary_pos_emb), attention_mask)
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
                logits = parallel_lm_logits(
                    hidden_states,
                    embedding.word_embeddings_weight,
                    self.parallel_output)
                return logits, presents
            else:
                logits = parallel_lm_logits(
                    lm_output,
                    embedding.word_embeddings_weight,
                    self.parallel_output)
                return logits

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
            self.specs.append(
                LayerSpec(
                    ParallelLinearPipe,
                    parallel_output=self.parallel_output
                )
            )
        # so output in training should just be logits
        # in inference it will be (logits, presents) (assuming get_key_value) is true

    def to_sequential(self):
        """
        Transforms the PipelineModule to a plain nn.Sequential module
        :return:
        """
        layers = []
        from collections import defaultdict
        tied_layers = defaultdict(list)
        for n, spec in enumerate(self.specs):
            if isinstance(spec, TiedLayerSpec):
                if spec.key in tied_layers:
                    # receiver
                    layers.append(Lambda(lambda x: spec.forward_fn(tied_layers[spec.key][0], x)))
                else:
                    # owner
                    module = spec.build(log=False)
                    layers.append(module)
                    tied_layers[spec.key].append(module)
            elif isinstance(spec, LayerSpec):
                layers.append(spec.build(log=False))
            else:
                # check that it's a lambda function
                LAMBDA = lambda:0
                if isinstance(spec, type(LAMBDA)) and spec.__name__ == LAMBDA.__name__:
                    # we assume it is a lambda function
                    layers.append(Lambda(spec))
                else:
                    raise ValueError(f'Layer number {n} ({spec}) Not recognized')
        return torch.nn.Sequential(*layers)

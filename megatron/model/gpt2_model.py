# coding=utf-8
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
from megatron import mpu
from megatron.module import MegatronModule

from .language_model import parallel_lm_logits
from .language_model import get_language_model
from .utils import init_method_normal
from .utils import scaled_init_method_normal

# Pipeline parallelism
from megatron import mpu
import torch.nn.functional as F
import torch.nn.functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
import megatron.fp16 as fp16
from megatron.model.transformer import ParallelTransformerLayerPipe
from .language_model import EmbeddingPipe

from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec


def gpt2_attention_mask_func(attention_scores, ltor_mask):
    attention_scores.masked_fill_(ltor_mask, -10000.0)
    return attention_scores


def CrossEntropy(output, labels):
    """ From pretrain_gpt2:forward_step() """
    labels, loss_mask = labels[0], labels[1]

    losses = mpu.vocab_parallel_cross_entropy(output.contiguous().float(), labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss



class GPT2Model(MegatronModule):
    """GPT-2 Language model."""

    def __init__(self, num_tokentypes=0, parallel_output=True):
        super(GPT2Model, self).__init__()
        args = get_args()

        self.parallel_output = parallel_output
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy

        self.language_model, self._language_model_key = get_language_model(
            attention_mask_func=gpt2_attention_mask_func,
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            init_method=init_method_normal(args.init_method_std),
            scaled_init_method=scaled_init_method_normal(args.init_method_std,
                                                         args.num_layers))

    def forward(self, input_ids, position_ids, attention_mask, labels=None,
                tokentype_ids=None, layer_past=None, get_key_value=False,
                forward_method_parallel_output=None):

        # Language model.
        lm_output = self.language_model(input_ids,
                                        position_ids,
                                        attention_mask,
                                        tokentype_ids=tokentype_ids,
                                        layer_past=layer_past,
                                        get_key_value=get_key_value)

        if get_key_value:
            lm_output, presents = lm_output

        # Output.
        parallel_output = self.parallel_output
        if forward_method_parallel_output is not None:
            parallel_output = forward_method_parallel_output
        output = parallel_lm_logits(
            lm_output,
            self.language_model.embedding.word_embeddings.weight,
            parallel_output)

        if get_key_value:
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


class GPT2ModelPipe(PipelineModule,MegatronModule):
    """GPT2Model adapted for pipeline parallelism.

    The largest change is flattening the GPTModel class so we can express it as a
    sequence of layers including embedding, transformer layers, and output.
    """

    def __init__(self, num_tokentypes=0, parallel_output=True, add_pooler=False, topology=None):
        args = get_args()

        self.parallel_output = parallel_output
        self.hidden_size = args.hidden_size
        self.num_tokentypes = num_tokentypes
        self.init_method = init_method_normal(args.init_method_std)
        self.output_layer_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)
        self.add_pooler = add_pooler
        if self.add_pooler:
            raise NotImplementedError('Pipeline pooler not yet implemented. Forward needs pooling_sequence_index')
        
        # Use torch gelu unless otherwise forced.
        gelu = F.gelu
        if args.openai_gelu:
            gelu = openai_gelu

        #
        # forward() prototype
        # 
        self.specs = []

        # Embedding layer
        self.specs.append(TiedLayerSpec('embed',
                                        EmbeddingPipe,
                                        self.hidden_size,
                                        args.padded_vocab_size,
                                        args.max_position_embeddings,
                                        args.hidden_dropout,
                                        self.init_method,
                                        self.num_tokentypes,
                                        tied_weight_attr='word_embeddings_weight'))

        # outputs are now (hidden_states, attention_mask)

        # data format change to avoid explicit tranposes : [b s h] --> [s b h]
        self.specs.append(lambda x: (x[0].transpose(0,1).contiguous(), x[1]))

        # Transformer layers
        for x in range(args.num_layers):
            self.specs.append(
                LayerSpec(ParallelTransformerLayerPipe,
                          attention_mask_func=gpt2_attention_mask_func,
                          init_method=self.init_method,
                          output_layer_init_method=self.output_layer_init_method,
                          layer_number=x))
        # Undo data format change and drop mask
        self.specs.append(lambda x: x[0].transpose(0,1).contiguous())


        # Final layernorm after transformer layers
        self.specs.append(
            LayerSpec(LayerNorm,
                      args.hidden_size,
                      eps=args.layernorm_epsilon))

        # XXX forward_method_parallel_output is assumed to be None, but we're not in a
        # fwd method to assert

        def _logits_helper(embedding, lm_output):
            """Just a wrapper to massage inputs/outputs from pipeline. """
            return parallel_lm_logits(
                lm_output,
                embedding.word_embeddings_weight,
                self.parallel_output)

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

        # Should maybe be done in loss_fn() instead?
        if args.fp16:
            self.specs.append(fp16.fp16_to_fp32)

        if args.checkpoint_activations:
            interval = args.checkpoint_num_layers
        else:
            interval = 0
        super().__init__(layers=self.specs,
                         loss_fn=CrossEntropy,
                         topology=topology,
                         activation_checkpoint_interval=interval,
                         partition_method='type:transformer')

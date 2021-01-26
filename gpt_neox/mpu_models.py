#copied from Sid's MegatronPipeline

# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
import torch.nn.functional as F
from deepspeed.pipe import PipelineModule, LayerSpec
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
import math

import mpu


def init_method_normal(std=0.02):
    """Init method based on normal distribution.

    This is only used for embeddings. The transformer has its
    own initializer.
    """
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)
    return init_

def scaled_init_method(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


class EmbedBlock(torch.nn.Module):

    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_sequence_length,
        embedding_dropout_prob,
        init_method_std=0.02
    ):
        super(EmbedBlock, self).__init__()
        init_method = init_method_normal(std=init_method_std)                            
        # Word embeddings (parallel).
        self.word_embeddings = mpu.VocabParallelEmbedding(
            vocab_size, hidden_size, init_method=init_method)

        # Position embedding (serial).
        self.position_embeddings = torch.nn.Embedding(max_sequence_length,
                                                      hidden_size)
        
        # Initialize the position embeddings.
        init_method(self.position_embeddings.weight)

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

    def forward(self, inputs):
        input_ids, position_ids, attention_mask = inputs
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings

        # Dropout.
        embeddings = self.embedding_dropout(embeddings)
        return (embeddings, attention_mask)


class GPT2Model(torch.nn.Module):
    """GPT-2 Language model.

    The output of the forward method are the logits (parallel or
    serial depending on the `parallel_output` flag.
    """

    def __init__(self,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 max_sequence_length,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 parallel_output=True):

        super(GPT2Model, self).__init__()

        self.parallel_output = parallel_output

        init_method = init_method_normal(std=0.02)

        # Word embeddings (parallel).
        self.word_embeddings = mpu.VocabParallelEmbedding(
            vocab_size, hidden_size, init_method=init_method)

        # Position embedding (serial).
        self.position_embeddings = torch.nn.Embedding(max_sequence_length,
                                                      hidden_size)

        # Initialize the position embeddings.
        init_method(self.position_embeddings.weight)

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

        # Transformer
        self.transformer = mpu.GPT2ParallelTransformer(num_layers,
                                                       hidden_size,
                                                       num_attention_heads,
                                                       attention_dropout_prob,
                                                       output_dropout_prob,
                                                       checkpoint_activations,
                                                       checkpoint_num_layers)
        
        self.seq_len = max_sequence_length

    def forward(self, input_ids, position_ids, attention_mask):

        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings

        # Dropout.
        embeddings = self.embedding_dropout(embeddings)

        # Transformer.
        transformer_output = self.transformer(embeddings, attention_mask)

        # Parallel logits.
        transformer_output_parallel = mpu.copy_to_model_parallel_region(
            transformer_output)
        logits_parallel = F.linear(transformer_output_parallel,
                                   self.word_embeddings.weight)

        if self.parallel_output:
            return logits_parallel

        return mpu.gather_from_model_parallel_region(logits_parallel)


class GPT2PipelineParallelTransformerLayer(mpu.transformer.GPT2ParallelTransformerLayer):
    def forward(self, inputs):
        hidden, mask = inputs
        outputs = super().forward(hidden, mask)
        return (outputs, mask)

class PipelineLayerNorm(torch.nn.Module):

    def __init__(self, hidden_size, layernorm_epsilon):
        super(PipelineLayerNorm, self).__init__()
        self.layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

    def forward(inputs):
        hidden, mask = inputs
        outputs = fn(hidden)
        return (outputs, mask)

class PipelineLinear(torch.nn.Module):

    def __init__(
        self,
        hidden_size,
        vocab_size,
        parallel_output=True
    ):
        super(PipelineLinear, self).__init__()
        self.linear = torch.nn.Linear(hidden_size, vocab_size)
        self.parallel_output = parallel_output

    def forward(self, inputs):
        # Parallel logits.
        transformer_output_parallel = mpu.copy_to_model_parallel_region(inputs)
        logits_parallel = self.linear(transformer_output_parallel)

        if self.parallel_output:
            return logits_parallel

        return mpu.gather_from_model_parallel_region(logits_parallel)

class GPT2PipelineModel(PipelineModule):

    def __init__(self,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 max_sequence_length,
                 checkpoint_activations,
                 loss_fn,
                 num_stages=2,
                 checkpoint_num_layers=1,
                 layernorm_epsilon=1.0e-5,
                 init_method_std=0.02,
                 parallel_output=True,
                 use_scaled_init_for_output_weights=True,
                 **kwargs):

        output_layer_init_method = None
        if use_scaled_init_for_output_weights:
            output_layer_init_method = scaled_init_method(init_method_std,
                                                          num_layers)
        init_method = init_method_normal(init_method_std)

        # Build spec list
        # Input Embedding
        spec = [
            LayerSpec(EmbedBlock, vocab_size=vocab_size, 
                      hidden_size=hidden_size, max_sequence_length=max_sequence_length,
                      init_method_std=init_method_std, embedding_dropout_prob=embedding_dropout_prob)
        ]
        # Transformer layers
        for i in range(num_layers):
            spec.append(
                LayerSpec(
                    GPT2PipelineParallelTransformerLayer,
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_dropout_prob=attention_dropout_prob,
                    output_dropout_prob=output_dropout_prob,
                    init_method=init_method,
                    output_layer_init_method=output_layer_init_method,
                    layernorm_epsilon=layernorm_epsilon
                )
            )
        #Output norm and Linear
        spec += [
            LayerSpec(PipelineLayerNorm, hidden_size, layernorm_epsilon),
            LayerSpec(PipelineLinear, hidden_size, vocab_size),
            lambda x: x.transpose(1, 2)
        ]
        print(spec)
        assert len(spec) % num_stages == 0, f"for optimal performance, depth + 4 ({len(spec)}) should be divisible by the number of pipeline stages ({num_stages})"
        super().__init__(layers=spec, loss_fn=loss_fn, num_stages=num_stages, **kwargs)


def gpt2_get_params_for_weight_decay_optimization(module):

    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module_ in module.modules():
        if isinstance(module_, (mpu.LayerNorm, torch.nn.LayerNorm)):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                 if p is not None])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n == 'bias'])

    return weight_decay_params, no_weight_decay_params



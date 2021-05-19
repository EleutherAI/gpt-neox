import math
import torch
import torch.nn.functional as F

from .norms import LayerNorm, RMSNorm, ScaleNorm, get_norm
from megatron import mpu
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.activations import get_activation
from megatron.model.utils import exists
from megatron.model.positional_embeddings import RotaryEmbedding, apply_rotary_pos_emb
from megatron.model.fused_bias_dropout import get_bias_dropout_add, bias_dropout_add_fused_train, \
    bias_dropout_add_fused_inference
from megatron.model.utils import configure_sparse_attention
import torch
from collections import defaultdict

from functools import partial
from megatron.model.utils import Lambda, SequentialWrapper
from megatron.model.norms import get_norm
from megatron.model.init_functions import get_init_methods

from megatron import mpu
from megatron.mpu import ParallelRelativePositionBias
import megatron.fp16 as fp16
from megatron.model.transformer import ParallelTransformerLayerPipe, NormPipe, ParallelLinearPipe, parallel_lm_logits
from megatron.model.word_embeddings import EmbeddingPipe
# Pipeline parallelism
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec

class TinyAttention(torch.nn.Module):
    def __init__(self, neox_args, d_attn, d_ff, mask_fn):
        super().__init__()
        self.proj_qkv = torch.nn.Linear(d_ff*2, 3 * d_attn)
        self.scale = d_attn ** -0.5
        self.seq_len = neox_args.seq_length
        self.proj_ffn = torch.nn.Linear(d_attn, d_ff)
        self.softmax = FusedScaleMaskSoftmax(
                input_in_fp16 = neox_args.precision == "fp16",
                upper_triang_mask_fusion = neox_args.scaled_upper_triang_masked_softmax_fusion,
                general_mask_fusion = neox_args.scaled_masked_softmax_fusion,
                mask_func = mask_fn,
                softmax_in_fp32 = neox_args.attention_softmax_in_fp32,
                scale = None)

    def forward(self, x, attention_mask):
        q, k, v = torch.chunk(self.proj_qkv(x), 3, dim=-1)
        w = torch.einsum("bnd,bmd->bnm", q, k).unsqueeze(1) * self.scale
        a = self.softmax(w, mask=attention_mask[...,:self.seq_len,:self.seq_len]).squeeze(1)
        x = torch.einsum("bnm,bmd->bnd", a, v)
        return self.proj_ffn(x)


class SpatialGatingUnit(torch.nn.Module):
    def __init__(self, neox_args, d_ff, causal=True, mask_fn=None):
        super().__init__()
        self.causal = causal # default to true bc mlm btfo
        norm, eps = get_norm(neox_args)
        self.norm = norm(d_ff, eps=eps)
        self.proj = torch.nn.Linear(neox_args.seq_length, neox_args.seq_length)
        self.use_attn = neox_args.gmlp_attn_dim is not None
        if self.use_attn:
            assert mask_fn is not None
            self.attn = TinyAttention(neox_args=neox_args, d_attn=neox_args.gmlp_attn_dim, d_ff=d_ff, mask_fn=mask_fn)
        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.constant_(self.proj.bias, 1.)
    
    def forward(self, x, attention_mask):
        res, gate = x.chunk(2, dim=-1) # split along dim
        gate = self.norm(gate)
        if self.causal:
            mask = torch.ones(self.proj.weight.shape[:2], device=gate.device).triu_(1).bool()
            weight = self.proj.weight.masked_fill(mask, 0.)
        gate = F.linear(gate.transpose(2, 1), weight, self.proj.bias).transpose(2,1)
        if self.use_attn:
            gate = gate + self.attn(x, attention_mask)
        return gate * res


class GMLPBlock(torch.nn.Module):
    def __init__(self, neox_args, init_method, output_layer_init_method, ff_mult=4, causal=True, mask_fn=None):
        super().__init__()
        ff_dim = neox_args.hidden_size * ff_mult
        norm, eps = get_norm(neox_args)
        self.norm = norm(neox_args.hidden_size, eps=eps)
        self.input_linear = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=ff_dim * 2,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True)
        self.activation_func = get_activation(neox_args)
        ff_dim_parallel = mpu.divide(ff_dim, mpu.get_model_parallel_world_size())
        self.sgu = SpatialGatingUnit(neox_args, ff_dim_parallel, causal=True, mask_fn=mask_fn)
        self.output_linear = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=ff_dim,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True)
    
    def forward(self, args):
        if len(args) == 1:
            x = args
            attention_mask = None
        if len(args) == 2:
            x, attention_mask = args
        else:
            raise ValueError
        x = self.norm(x)
        x, _ = self.input_linear(x)
        x = self.activation_func(x)
        x = self.sgu(x, attention_mask)
        x, _ = self.output_linear(x)
        return x, attention_mask
        

class GMLPModelPipe(PipelineModule, torch.nn.Module):
    """Autoregressive 3D parallel GMLPModel."""

    def __init__(self, neox_args, num_tokentypes=0, parallel_output=True, topology=None, inference=False):
        from megatron.model.gpt2_model import gpt2_attention_mask_func, cross_entropy
        self.neox_args = neox_args
        self._inference = inference
        self.parallel_output = parallel_output
        self.hidden_size = self.neox_args.hidden_size
        self.num_tokentypes = num_tokentypes
        self.init_method, self.output_layer_init_method = get_init_methods(self.neox_args)
        self.attention_mask_func = gpt2_attention_mask_func
        self.specs = []
        self.init_specs()
        loss_fn = partial(cross_entropy, _fp16=self.neox_args.fp16_lm_cross_entropy)
        if self.neox_args.checkpoint_activations:
            interval = self.neox_args.checkpoint_num_layers
        else:
            interval = 0
        super().__init__(layers=self.specs,
                         loss_fn=loss_fn if not self._inference else None,
                         topology=topology,
                         activation_checkpoint_interval=interval,
                         partition_method='type:GMLPBlock',
                         checkpointable_layers='GMLPBlock')

    def init_specs(self):
        self.specs = []
        # Embedding layer
        if not self.neox_args.no_weight_tying:
            self.specs.append(TiedLayerSpec('embed',
                                            EmbeddingPipe,
                                            neox_args = self.neox_args,
                                            hidden_size = self.hidden_size,
                                            vocab_size = self.neox_args.padded_vocab_size,
                                            max_sequence_length = self.neox_args.max_position_embeddings,
                                            embedding_dropout_prob = self.neox_args.hidden_dropout,
                                            init_method = self.init_method,
                                            num_tokentypes = self.num_tokentypes,
                                            use_pos_emb = False,
                                            tied_weight_attr='word_embeddings_weight'))
        else:
            self.specs.append(LayerSpec(EmbeddingPipe,
                                        neox_args = self.neox_args,
                                        hidden_size = self.hidden_size,
                                        vocab_size = self.neox_args.padded_vocab_size,
                                        max_sequence_length = self.neox_args.max_position_embeddings,
                                        embedding_dropout_prob = self.neox_args.hidden_dropout,
                                        init_method = self.init_method,
                                        num_tokentypes = self.num_tokentypes,
                                        use_pos_emb = False))

        self.specs.append(lambda x: (x[0].contiguous(), *x[1:]))

        # GMLP layers
        for x in range(self.neox_args.num_layers):
            self.specs.append(
                LayerSpec(
                    GMLPBlock,
                    init_method=self.init_method,
                    output_layer_init_method=self.output_layer_init_method,
                    neox_args=self.neox_args,
                    mask_fn=self.attention_mask_func
                )
            )
            
        # drop mask
        self.specs.append(lambda x: x[0].contiguous())

        norm, eps = get_norm(self.neox_args)
        self.specs.append(
            LayerSpec(NormPipe,
                      norm,
                      self.neox_args.hidden_size,
                      eps=eps))

        def _logits_helper(embedding, lm_output):
            """Just a wrapper to massage inputs/outputs from pipeline. """
            logits = parallel_lm_logits(
                    lm_output,
                    embedding.word_embeddings_weight,
                    self.parallel_output)
            return logits

        if not self.neox_args.no_weight_tying:
            self.specs.append(
                TiedLayerSpec('embed',
                              EmbeddingPipe,
                              neox_args = self.neox_args,
                              hidden_size = self.hidden_size,
                              vocab_size = self.neox_args.padded_vocab_size,
                              max_sequence_length = self.neox_args.max_position_embeddings,
                              embedding_dropout_prob = self.neox_args.hidden_dropout,
                              init_method = self.init_method,
                              num_tokentypes = self.num_tokentypes,
                              use_pos_emb = False,
                              forward_fn = _logits_helper,
                              tied_weight_attr='word_embeddings_weight')
            )
        else:
            self.specs.append(
                LayerSpec(
                    ParallelLinearPipe,
                    neox_args=self.neox_args,
                    init_method=self.init_method,
                    parallel_output=self.parallel_output
                )
            )

    def to_sequential(self):
        """
        Transforms the PipelineModule to a plain nn.Sequential module
        :return:
        """
        layers = []
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
            elif hasattr(spec, '__call__'):
                # check that it's a callable function
                layers.append(Lambda(spec))
            else:
                raise ValueError(f'Layer number {n} ({spec}) Not recognized')
        model = SequentialWrapper(layers,
                                  self.activation_checkpoint_interval,
                                  self.activation_checkpoint_func,
                                  parent_class_name=self.__class__.__name__)
        return model


from megatron.model.gpt2_model import *
from megatron.model.norms import get_norm_and_eps


class GPT2Momentum(torch.nn.Module):

    def __init__(self, neox_args, num_tokentypes=0, parallel_output=True, topology=None, inference=False,
                 get_key_value=True):
        self.neox_args = neox_args

        self._inference = inference
        self.get_key_value = get_key_value if inference else False
        self.parallel_output = parallel_output
        self.hidden_size = self.neox_args.hidden_size
        self.num_tokentypes = num_tokentypes
        self.init_method, self.output_layer_init_method = get_init_methods(self.neox_args)
        self.fp16_lm_cross_entropy = self.neox_args.fp16_lm_cross_entropy
        self.embedding_type = self.neox_args.pos_emb
        if self.embedding_type == 'rpe':
            raise NotImplementedError

        self.embedding = EmbeddingPipe(
            neox_args=self.neox_args,
            hidden_size=self.hidden_size,
            vocab_size=self.neox_args.padded_vocab_size,
            max_sequence_length=self.neox_args.max_position_embeddings,
            embedding_dropout_prob=self.neox_args.hidden_dropout,
            init_method=self.init_method
        )

        # NormPipe is a helper class to pass presents through to the output when doing inference
        norm, eps = get_norm_and_eps(neox_args)
        self.final_layernorm = NormPipe(norm,
                                        self.neox_args.hidden_size,
                                        eps=eps)
        if neox_args.no_weight_tying:
            self.final_linear = ParallelLinearPipe(
                neox_args=self.neox_args,
                init_method=self.init_method,
                parallel_output=self.parallel_output)

        self.n_iters = 12  # number of momentum iterations - hardcode to 12 for now
        self.gamma = 0.99 # gamma for momentum - hardcode to .99 for now
        for idx, layer in enumerate(self.get_transformer_layers()):
            self.add_module(f"transformer_block_{i}", layer)
        # TODO: activation checkpointing
        # if self.neox_args.checkpoint_activations:
        #     interval = self.neox_args.checkpoint_num_layers
        # else:
        #     interval = 0

    def get_transformer_layers(self):
        layers = []

        # Transformer layers
        for x in range(self.neox_args.num_layers):
            layers.append(ParallelTransformerLayerPipe(
                neox_args=self.neox_args,
                attention_mask_func=gpt2_attention_mask_func,
                init_method=self.init_method,
                output_layer_init_method=self.output_layer_init_method,
                layer_number=x,
                rpe=None,
                rotary=self.neox_args.pos_emb == 'rotary',
                get_key_value=self.get_key_value
            )
            )

        return layers

    def to_logits(self, lm_output):
        """Just a wrapper to massage inputs/outputs from pipeline. """
        if self.neox_args.no_weight_tying:
            return self.final_linear(lm_output)
        else:
            if self._inference and len(lm_output) == 2:
                hidden_states, presents = lm_output
                logits = parallel_lm_logits(
                    hidden_states,
                    self.embedding.word_embeddings_weight,
                    self.parallel_output)
                return logits, presents
            else:
                logits = parallel_lm_logits(
                    lm_output,
                    self.embedding.word_embeddings_weight,
                    self.parallel_output)
                return logits

    def forward(self, x, ts=1):
        x = self.embedding(x)
        if self._inference:
            # we need to add a container to cache `presents` from each layer's forward pass
            # inputs/outputs are now (hidden_states, layer_past, presents, attention_mask)
            x = (x[0].transpose(0, 1).contiguous(), x[1], torch.Tensor(), *x[2:])
        else:
            x = (x[0].transpose(0, 1).contiguous(), *x[1:])
        v = torch.zeros_like(x[0])
        for _ in range(self.n_iters):
            for layer in self.transformer_layers:
                x = layer(x)
                v = self.gamma * v + x[0] * ts * (1 - self.gamma)
                x[0] = x[0] + v * ts
        if self._inference:
            # from (hidden_states, layer_past, presents, attention_mask)
            # to (hidden_states^T, presents)
            x = (x[0].transpose(0, 1).contiguous(), x[2])
        else:
            # Undo data format change and drop mask
            x = x[0].transpose(0, 1).contiguous()
        logits = self.to_logits(x)
        if self._inference:
            logits, presents = logits
            return logits
        return logits

    @property
    def transformer_layers(self):
        return [self._modules[f"transformer_block_{i}"] for i in range(self.neox_args.num_layers)]

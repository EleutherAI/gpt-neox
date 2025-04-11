import torch
from torch import nn
from transformers import GPTNeoXForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXMLP


class TransformerEngineGPTNeoXMLP(GPTNeoXMLP):
    """Custom MLP with LayerNorm for Transformer Engine compatibility"""

    def __init__(self, config):
        super().__init__(config)
        # Add the LayerNorm that Transformer Engine uses
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        # Apply layer norm before the dense_h_to_4h (FC1)
        hidden_states = self.layer_norm(hidden_states)
        # Then proceed with the original MLP forward pass
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


class TransformerEngineGPTNeoXForCausalLM(GPTNeoXForCausalLM):
    """GPTNeoX with custom MLP that includes LayerNorm for Transformer Engine compatibility"""

    def __init__(self, config):
        super().__init__(config)
        # Replace all MLP instances with our custom MLP
        for i, layer in enumerate(self.gpt_neox.layers):
            self.gpt_neox.layers[i].mlp = TransformerEngineGPTNeoXMLP(config)

    @classmethod
    def from_config(cls, config):
        """Creates a TransformerEngineGPTNeoXForCausalLM from a configuration."""
        return cls(config)

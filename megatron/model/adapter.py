import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import Literal

class Adapter(nn.Module):
    def __init__(
        self,
        dim: int,
        downsample_factor: int = 4,
        activation: nn.Module = nn.ReLU,
        add_layernorm: bool = False,
    ):
        super().__init__()
        layers = []
        if add_layernorm:
            layers.append(nn.LayerNorm(dim))
        layers.extend(
            [
                nn.Linear(dim, dim // downsample_factor),
                activation(),
                nn.Linear(dim // downsample_factor, dim),
            ]
        )
        self.adapter = nn.Sequential(*layers)
        self.adapter.apply(self.init_weights)

    def init_weights(self, m: nn.Module, std=1e-3):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=std)
            torch.nn.init.normal_(m.bias, std=std)
            m.weight.data = torch.clamp(m.weight.data, min=-2 * std, max=2 * std)
            m.bias.data = torch.clamp(m.bias.data, min=-2 * std, max=2 * std)
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)

    def forward(self, x: TensorType["b", "s", "d"]) -> TensorType["b", "s", "d"]:
        return self.adapter(x) + x


class ParallelAdapter(Adapter):
    def __init__(
        self,
        module: nn.Module,
        dim: int,
        downsample_factor: int = 4,
        scaled: bool = False,
        add_layernorm: bool = False,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__(
            dim, downsample_factor, add_layernorm=add_layernorm, activation=activation
        )
        self.module = module

        if scaled:
            # init scaling param
            self.adapter_scale = nn.Parameter(torch.ones(1))
        else:
            self.adapter_scale = 1

    def forward(self, x: TensorType["b", "s", "d"], **module_kwargs):
        y = self.module(x, **module_kwargs)
        z = self.adapter(x)
        return y + (z * self.adapter_scale)


class ParallelAdapterWrapper(ParallelAdapter):
    # used to add an adapter to the attention block

    def __init__(
        self,
        module: nn.Module,
        dim: int,
        downsample_factor: int = 4,
        scaled: bool = False,
        add_layernorm: bool = False,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__(
            module, dim, downsample_factor, scaled, add_layernorm, activation
        )

    def forward(self, x: TensorType["b", "s", "d"], *attn_args, **attn_kwargs):
        attn_outputs = self.module(x, *attn_args, **attn_kwargs)
        attn_output, outputs = (
            attn_outputs[0],
            attn_outputs[1:],
        )  # output_attn: a, present, (attentions)
        hidden_states = attn_output + (self.adapter(x) * self.adapter_scale)
        return (hidden_states,) + outputs


class AdapterWrapper(Adapter):
    # used to add an adapter to the attention block

    def __init__(
        self,
        attn_block: nn.Module,
        dim: int,
        downsample_factor: int = 4,
        activation: nn.Module = nn.ReLU,
        add_layernorm: bool = False,
    ):
        super().__init__(dim, downsample_factor, activation, add_layernorm)
        self.attn_block = attn_block

    def forward(self, x: TensorType["b", "s", "d"], *attn_args, **attn_kwargs):
        attn_outputs = self.attn_block(x, *attn_args, **attn_kwargs)
        
        attn_output, outputs = (
            attn_outputs[0],
            attn_outputs[1:],
        )  # outputs: output, bias
        hidden_states = self.adapter(attn_output) + attn_output
        return (hidden_states,) + outputs

def add_adapters(
        neox_args,
        model,
        downsample_factor: int = 4,
        # adapter_type: Literal["normal", "parallel", "scaled_parallel"] = "normal",
        location: Literal["mlp", "attention"] = "mlp",
        ff_attr: str = "mlp",
        attn_attr: str = "attention",
        **adapter_kwargs,    
):
    for names, module in model.named_modules():
        if 'image_prefix' in names:
          continue
          # avoid add adapter in image_prefix, may need modification when change the image_prefixer
        names = [name for name,module in module.named_modules()]
        if location in names and location==ff_attr:
            mlp = getattr(module,ff_attr)
            adapter_layer = AdapterWrapper(
                        attn_block=mlp,
                        dim=neox_args.hidden_size,
                        downsample_factor=downsample_factor,
                        **adapter_kwargs
                        )
            setattr(module,ff_attr,adapter_layer)   
        elif location in names and location==attn_attr:
            attn = getattr(module,attn_attr)
            adapter_layer = AdapterWrapper(
                        attn_block=attn,
                        dim=neox_args.hidden_size,
                        downsample_factor=downsample_factor,
                        **adapter_kwargs
                        )
            setattr(module,attn_attr,adapter_layer)          
    return model                 

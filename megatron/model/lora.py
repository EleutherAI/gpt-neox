
import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import Literal
import math

class Lora(nn.Module):
    def __init__(
        self,
        rank: int,
        in_features: int,
        out_features: int,
        activation: nn.Module = nn.ReLU,#Keeping this around for now
    ):
        super().__init__()

        self.lora_A = nn.Parameter(torch.zeros((rank, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, rank)))
        self.init_weights()
        self.lora_dropout = lambda x: x#empty for now
        self.lora_alpha = 1
        self.scaling = self.lora_alpha / rank
        
        #Do we want a learned bias also?

    def init_weights(self):
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        
    def lora(self, x):
        return (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        
    def forward(self, x: TensorType["b", "s", "d"]) -> TensorType["b", "s", "d"]:
        return self.lora(x) + x

class ParallelLora(Lora):
    def __init__(
        self,
        rank: int,
        module: nn.Module,
        in_features: int,
        out_features: int,
        activation: nn.Module = nn.ReLU,#Keeping this around for now
        scaled: bool = False,
    ):
        super().__init__(
            rank = rank, in_features = in_features, out_features = out_features, activation=activation
        )
        
        self.module = module

        if scaled:
            # init scaling param
            self.lora_scale = nn.Parameter(torch.ones(1))
        else:
            self.lora_scale = 1

    def forward(self, x: TensorType["b", "s", "d"], **module_kwargs):
        y = self.module(x, **module_kwargs)
        z = self.lora(x)
        return y + (z * self.lora_scale)


class ParallelLoraWrapper(ParallelLora):
    # used to add a lora to the attention block

    def __init__(
        self,
        rank: int,
        module: nn.Module,
        in_features: int,
        out_features: int,
        activation: nn.Module = nn.ReLU,#Keeping this around for now
    ):
        super().__init__(
            module = module, rank = rank, in_features = in_features, out_features = out_features, activation=activation
        )

    def forward(self, x: TensorType["b", "s", "d"], *attn_args, **attn_kwargs):
        attn_outputs = self.module(x, *attn_args, **attn_kwargs)
        attn_output, outputs = (
            attn_outputs[0],
            attn_outputs[1:],
        )  # output_attn: a, present, (attentions)
        hidden_states = attn_output + (self.lora(x) * self.lora_scale)
        return (hidden_states,) + outputs


class LoraWrapper(Lora):
    # used to add an lora to the attention block

    def __init__(
        self,
        attn_block: nn.Module,
        rank: int,
        in_features: int,
        out_features: int,
        activation: nn.Module = nn.ReLU,#Keeping this around for now
    ):
        super().__init__(rank = rank, in_features = in_features, out_features = out_features, activation = activation)
        self.attn_block = attn_block

    def forward(self, x: TensorType["b", "s", "d"], *attn_args, **attn_kwargs):
        attn_outputs = self.attn_block(x, *attn_args, **attn_kwargs)
        
        attn_output, outputs = (
            attn_outputs[0],
            attn_outputs[1:],
        )  # outputs: output, bias
        hidden_states = self.lora(attn_output) + attn_output
        return (hidden_states,) + outputs

def add_lora(
        neox_args,
        model,
        downsample_factor = 4, #Default downsample for adapter is 4x
        # adapter_type: Literal["normal", "parallel", "scaled_parallel"] = "normal",
        location: Literal["mlp", "attention"] = "mlp",
        ff_attr: str = "mlp",
        attn_attr: str = "attention",
        **lora_kwargs,    
):
    for names, module in model.named_modules():
        if 'image_prefix' in names:
            continue # no lora for image_prefix transformers
        names = [name for name, module in module.named_modules()]
        if location in names and location==ff_attr:
            mlp = getattr(module,ff_attr)
            lora_layer = LoraWrapper(
                        attn_block = mlp,
                        rank = neox_args.hidden_size//4,
                        in_features = neox_args.hidden_size,
                        out_features = neox_args.hidden_size,
                        **lora_kwargs
                        )
            setattr(module,ff_attr,lora_layer)   
        elif location in names and location==attn_attr:
            attn = getattr(module,attn_attr)
            lora_layer = LoraWrapper(
                        attn_block=attn,
                        rank = neox_args.hidden_size//4,
                        in_features = neox_args.hidden_size,
                        out_features = neox_args.hidden_size,
                        **lora_kwargs
                        )
            setattr(module,attn_attr,lora_layer)          
    return model

import torch
import torch.nn as nn

import math
import einops

from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

# TODO: make imports conditional
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

# TODO: put notation up here

# Mamba layer, without parallelism.
class MambaBlock(nn.Module):
    def __init__(
        self,
        neox_args,
        # ...
    ):
        super().__init__()

        self.neox_args

        dtype = (
            torch.float16 if neox_args.precision == "fp16" else torch.bfloat16
        )  # TODO: allow for fp32?
        factory_kwargs = {"device": torch.cuda.current_device(), "dtype": dtype}

        self.d_model = neox_args.hidden_size
        self.d_state = 16  # neox_args.mamba_state_dim
        self.d_conv = 4  # neox_args.mamba_conv_dim
        self.expand = 2
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(
            self.d_model / 16
        )  # if not neox_args.mamba_dt_rank else neox_args.mamba_dt_rank

        self.dt_init = "random"
        self.dt_min, self.dt_max, self.dt_init_floor = 0.001, 0.1, 1e-4
        assert self.dt_init in ["constant", "random"]

        self.use_bias = False  # TODO: add arg for this

        # up-projection in MambaBlock.
        self.in_proj = nn.Linear(
            self.d_model,
            self.d_inner * 2,
            bias=self.use_bias,
            **factory_kwargs,
        )

        # convolution in MambaBlock
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=self.use_bias,  # TODO: separate from Linear proj biases
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            **factory_kwargs,
        )

        self.act_fn = nn.SiLU()

        # x_proj corresponds to s_B(x), s_C(x), s_Delta(x)
        # in https://arxiv.org/pdf/2312.00752.pdf Algorithm 2
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )

        # up-project dt / Delta from dt_rank to d_inner
        self.dt_proj = nn.Linear(  # TODO: why does this use a bias?
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )

        # special init for dt_proj
        dt_init_std = (self.dt_rank**-0.5) * self.dt_scale
        if self.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif self.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # more dt_proj init stuff. copied from https://github.com/state-spaces/mamba/blob/009bec5ee37f586844a3fc89c040a9c1a9d8badf/mamba_ssm/modules/mamba_simple.py#L91-L101
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(self.dt_max) - math.log(self.dt_min))
            + math.log(self.dt_min)
        ).clamp(min=self.dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # initialize A . uses S4D real initialization #TODO: add a flag controlling this
        A = einops.repeat(
            torch.arange(
                1,
                self.d_state + 1,
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            ),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(
            A
        )  # important! # TODO: ensure DeepSpeed doesn't muck with A's dtype
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = (
            True  # TODO: ensure NeoX handles these weight decay properties right
        )

        # D parameter
        self.D = nn.Parameter(
            torch.ones(
                self.d_inner, device=torch.cuda.current_device(), dtype=torch.float32
            )
        )  # Important? Keep in fp32
        self.D._no_weight_decay = True  # TODO: same as A_log weight decay, see above

        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=self.use_bias, **factory_kwargs
        )

    def selective_scan(
        self,
        x,
        dt,
        A,
        B,
        C,
        D,
        z=None,
        delta_bias=None,
        delta_softplus=True,
    ):

        if not self.neox_args.mamba_fused_selective_scan:
            y = selective_scan_ref(
                u=x,
                delta=dt,
                A=A,
                B=B,
                C=C,
                D=D,
                z=z,
                delta_bias=delta_bias,
                delta_softplus=delta_softplus,
                return_last_state=False,
            )
        else:
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                D=D,
                z=z,
                delta_bias=delta_bias,
                delta_softplus=delta_softplus,
                return_last_state=False,
            )

        return y

    def forward(self, hidden_states):
        """ """
        # hidden_states: [sq, b, h]
        seqlen, batch, dim = hidden_states.shape

        # TODO: support inference in separate method
        # For now, only handle training (parallel scan).

        # first up: perform in_proj
        # TODO: make this compatible with TP / Megatron ParallelLinears
        xz = einops.rearrange(
            self.in_proj.weight @ einops.rearrange(hidden_states, "l b d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )  # TODO: is the fact hidden_states input is different shape from Mamba code hidden states shape a problem?

        if self.in_proj.bias is not None:
            xz = xz + einops.rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        # TODO: add use_fast_path equiv. from mamba code

        A = -torch.exp(
            self.A_log.float()
        )  # (d_inner, d_state) # TODO: move this to right before selective scan?

        x, z = xz.chunk(2, dim=1)

        # ===========
        # Convolution
        # ===========

        # TODO: use causal_conv1d cuda kernels, make configurable
        x = self.act_fn(self.conv1d(x)[..., :seqlen])

        # ==============
        # SSM Projection
        # ==============

        # project: perform s_B, s_C, s_Delta projections
        x_dbl = self.x_proj(einops.rearrange(x, "b d l -> (b l) d"))  # shape (bl d)
        # split into component dt, B, C
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # up-project Delta / dt
        dt = self.dt_proj.weight @ dt.t()
        dt = einops.rearrange(dt, "d (b l) -> b d l", l=seqlen)

        # rearrange B, C
        B = einops.rearrange(B, "(b l) d_state -> b d_state l", l=seqlen).contiguous()
        C = einops.rearrange(C, "(b l) d_state -> b d_state l", l=seqlen).contiguous()

        # TODO: assert activation is silu or swish for scan?

        y = self.selective_scan(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),  # TODO: why's this cast here?
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            # return_last_state, #TODO: inference concerns
        )

        # ===============
        # Down-Projection
        # ===============

        out = self.out_proj(y)

        return out


class MambaResidualLayer(nn.Module):
    """
    Pre-norm Mamba Block with residual connection. No parallelism yet supported.
    """

    def __init__(
        self,
        neox_args,
        layer_number,
    ):
        super().__init__()
        # TODO: allow for residual in fp32 if it helps?
        self.layer_number = layer_number

        norm, eps = get_norm(neox_args)

        self.norm = norm(neox_args, eps=eps)

        # TODO: dropout

        self.mixer = MambaBlock(
            neox_args=neox_args,
        )

    def forward(self, x, attention_mask=None, layer_past=None):

        # pseudocode:
        # x = x + mixer(norm(x))
        residual = x

        hidden_states = self.mixer(
            self.norm(x)
        )  # TODO: do norm in fp32? what is it by default

        return hidden_states + residual


class MambaResidualLayerPipe(MambaResidualLayer):
    """Extends MambaResidualLayer to forward attention_mask through the pipeline. DeepSpeed requires this."""

    def forward(self, args):
        assert (
            len(args) == 2
        ), "MambaResidualLayerPipe expects 2 arguments - hidden_states and attention_mask"
        hidden_states, attention_mask = args
        # we are returning just [hidden_states, mask]
        return super().forward(hidden_states, attention_mask), attention_mask

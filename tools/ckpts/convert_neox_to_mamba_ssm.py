import torch

from convert_neox_to_hf import load_partitions, get_key, get_state

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


import argparse
from typing import Literal
import yaml
from tqdm import tqdm

import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
    )
)
from megatron.tokenizer import build_tokenizer

"""
Conversion utility for converting a Mamba model
trained in GPT-NeoX into the mamba_ssm package ckpt format.
"""
ARCH = {
    "COLUMN_PARALLEL_LINEAR_KEYS": {
        # these require concat across dim=0
        "mixer.in_proj.weight": "mixer.in_proj.weight",
        # "mixer.in_proj.bias": "mixer.in_proj.bias",
        "mixer.A_log": "mixer.A_log",
        "mixer.D": "mixer.D",
        "mixer.conv1d.weight": "mixer.conv1d.weight",
        "mixer.conv1d.bias": "mixer.conv1d.bias",
        "mixer.dt_proj.weight": "mixer.dt_proj.weight",
        "mixer.dt_proj.bias": "mixer.dt_proj.bias",
    },
    "ROW_PARALLEL_LINEAR_KEYS": {
        # these require concat across dim=1
        "mixer.out_proj.weight": "mixer.out_proj.weight",
        "mixer.x_proj.weight": "mixer.x_proj.weight",
    },
    "ROW_PARALLEL_BIAS_KEYS": {
        # these require summing across ranks
        # "mixer.x_proj.bias": "mixer.x_proj.bias",
        # "mixer.out_proj.bias": "mixer.out_proj.bias",
    },
    "NORM_KEYS": {
        "norm.scale": "norm.weight",
        # "norm.bias": "norm.bias",
    },
    "FINAL_NORM_KEYS": {
        "norm.scale": "weight",
        # "norm.bias": "bias",
    },
}


def create_config(neox_config):
    class TokenizerArgs:
        # kinda hacky.
        # this is to get something with the same interface as is used in build_tokenizer()
        # without diving into loading a neox_args object or using argparse etc.
        def __init__(self, neox_config):
            self.make_vocab_size_divisible_by = get_key(
                neox_config, "make-vocab-size-divisible-by", default=128
            )
            self.model_parallel_size = get_key(neox_config, "model-parallel-size")
            self.vocab_file = get_key(neox_config, "vocab-file")
            self.merge_file = get_key(neox_config, "merge-file")
            self.tokenizer_type = get_key(neox_config, "tokenizer-type")

            self.rank = 0

    args = TokenizerArgs(neox_config)
    tokenizer = build_tokenizer(args)
    try:  # GPT2TokenizerFast raises NotImplementedError
        pad_token = tokenizer.pad
    except:
        pad_token = (
            1  # pad defaulting to 1. follows convention from GPT-NeoX-20b tokenizer
        )
    norm_type = get_key(neox_config, "norm", "layernorm")
    if norm_type == "rmsnorm":
        use_rms_norm = True
    else:
        assert (
            norm_type == "layernorm"
        ), "only layernorm or rmsnorm supported by mamba_ssm!"
        use_rms_norm = False
    return MambaConfig(
        d_model=get_key(neox_config, "hidden_size"),
        n_layer=get_key(neox_config, "num_layers"),
        vocab_size=args.padded_vocab_size,
        rms_norm=use_rms_norm,
        residual_in_fp32=False,
        fused_add_norm=True,
        # shouldn't really matter? we didn't train with it but should be equiv.
        # it's faster though
        # pad_vocab_size_multiple_of=get_key(neox_config, "make_vocab_size_divisible_by", 128),
        tie_embeddings=not get_key(
            neox_config, "no_weight_tying", False
        ),  # requires newer mamba_ssm>=1.2.0.post1
    )


def convert(
    input_checkpoint_path,
    loaded_config,
    output_checkpoint_path,
    sequential: bool = True,
    precision: Literal["auto", "fp16", "bf16", "fp32"] = "auto",
):

    mamba_config = create_config(loaded_config)

    if precision == "auto":
        print("Auto-detecting precision to save model into...")
        # save model in FP16 if Deepspeed fp16 was used in config, else 32 bit
        fp16 = get_key(loaded_config, "fp16")

        if fp16:
            try:
                # current behavior is to pass "fp16": {"enabled": true}, when using upstream Deepspeed
                if fp16["enabled"]:
                    dtype = torch.float16
                    print("Saving weights in fp16 precision...")
            except:
                try:
                    # attempt to access bf16 dict in yaml file, if fp16 not enabled
                    bf16 = get_key(loaded_config, "bf16")
                    if bf16:
                        dtype = torch.bfloat16
                        print("Saving weights in bf16 precision...")
                except:
                    dtype = torch.float
                    print(
                        "Model not trained in fp16 / bf16 mixed precision, saving weights in fp32..."
                    )
    else:
        name_to_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float,
        }
        print(f"Saving model into specified {precision} precision...")
        dtype = name_to_dtype[precision]

    mamba_model = MambaLMHeadModel(
        config=mamba_config,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float,
    )  # dtype)

    mp_partitions = get_key(loaded_config, "model-parallel-size")

    # Sequential saves all model states from an MP rank in one file.
    # so we only load the MP ranks only once and index into them with get_state().
    # for the pipeline-parallel case (pipeline-parallel-size >= 1),
    # we must load the correct layer's states at each step.
    # (this does mean that less memory is required for PP conversion.)
    loaded_tp_ranks = load_partitions(
        input_checkpoint_path, mp_partitions, layer_idx=0, sequential=sequential
    )

    mamba_model.backbone.embedding.load_state_dict(
        {
            "weight": torch.cat(
                get_state(
                    loaded_tp_ranks,
                    "word_embeddings.weight",
                    layer_idx=0,
                    sequential=sequential,
                ),
                dim=0,
            )
        }
    )

    for layer_i in tqdm(range(get_key(loaded_config, "num-layers"))):

        layer = mamba_model.backbone.layers[layer_i]

        if not sequential:
            # in the non-sequential case, must load from each layer individually.
            # use layer index + 2 bc of embed layer and a dummy _pre_transformer_block, which are "layers 0 and 1"
            loaded_tp_ranks = load_partitions(
                input_checkpoint_path,
                mp_partitions,
                layer_idx=layer_i + 2,
                sequential=sequential,
            )

        state_dict = {}

        for key, hf_key in ARCH["ROW_PARALLEL_LINEAR_KEYS"].items():  # ROW_PARALLEL
            state_dict[hf_key] = torch.cat(
                get_state(
                    loaded_tp_ranks, key, layer_idx=layer_i + 2, sequential=sequential
                ),
                dim=1,
            )

        # average layernorm stats over mp ranks
        for key, hf_key in ARCH["NORM_KEYS"].items():
            state_dict[hf_key] = sum(
                get_state(
                    loaded_tp_ranks, key, layer_idx=layer_i + 2, sequential=sequential
                )
            ) / len(loaded_tp_ranks)

        # LinearWithTPMerge
        for key, hf_key in ARCH["COLUMN_PARALLEL_LINEAR_KEYS"].items():
            state_dict[hf_key] = torch.cat(
                get_state(
                    loaded_tp_ranks, key, layer_idx=layer_i + 2, sequential=sequential
                ),
                dim=0,
            )

        # LinearWithTPSplitBias
        for key, hf_key in ARCH["ROW_PARALLEL_BIAS_KEYS"].items():
            state_dict[hf_key] = sum(
                get_state(
                    loaded_tp_ranks, key, layer_idx=layer_i + 2, sequential=sequential
                )
            )

        layer.load_state_dict(state_dict)

    if not sequential:
        loaded_tp_ranks = load_partitions(
            input_checkpoint_path,
            mp_partitions,
            get_key(loaded_config, "num-layers") + 3,
            sequential=sequential,
        )

    norm_state_dict = {}
    for key, hf_key in ARCH["FINAL_NORM_KEYS"].items():
        norm_state_dict[hf_key] = sum(
            get_state(
                loaded_tp_ranks,
                key,
                layer_idx=get_key(loaded_config, "num-layers") + 3,
                sequential=sequential,
            )
        ) / len(loaded_tp_ranks)

    final_layer_norm = mamba_model.backbone.norm_f

    final_layer_norm.load_state_dict(norm_state_dict)

    if not sequential:
        loaded_tp_ranks = load_partitions(
            input_checkpoint_path,
            mp_partitions,
            get_key(loaded_config, "num-layers") + 4,
            sequential=sequential,
        )

    lm_head = mamba_model.lm_head

    lm_head.load_state_dict(
        {
            "weight": torch.cat(
                get_state(
                    loaded_tp_ranks,
                    "final_linear.weight",
                    layer_idx=get_key(loaded_config, "num-layers") + 4,
                    sequential=sequential,
                ),
                dim=0,
            ),
        }
    )

    del loaded_tp_ranks

    return mamba_model


def main(input_args=None, overwrite_values=None):

    parser = argparse.ArgumentParser(
        description="Merge MP partitions and convert to HF Model."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to NeoX checkpoint, e.g. /path/to/model/global_step143000",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to config file for the input NeoX checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output dir, where to save the HF Model, tokenizer, and configs",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        help="What precision to save the model into. Defaults to auto, which auto-detects which 16-bit dtype to save into, or falls back to fp32.",
    )
    parser.add_argument(
        "--no_save_tokenizer",
        action="store_true",
        help="Whether to skip saving the tokenizer alongside a model.",
    )
    args = parser.parse_args(input_args)

    # validate arguments
    assert args.precision in [
        "auto",
        "fp16",
        "bf16",
        "fp32",
    ], f"expected --precision to be one of 'auto', 'fp16', 'bf16', 'fp32' but got '{args.precision}' !"

    with open(args.config_file) as f:
        loaded_config = yaml.full_load(f)
        if overwrite_values:
            loaded_config.update(overwrite_values)

    # Determine the checkpoint format of the model.
    # DeepSpeed saves models wrapped in a PipelineModule differently from those not.
    # PipelineModule models are saved as per-layer state dicts per TP shard,
    # while Sequential model state dicts are saved all together in one mp_rank_xx_model_states.pt
    # file per tensor/model parallel shard.
    pipeline_world_size = get_key(loaded_config, "pipe-parallel-size", 1)
    if pipeline_world_size == 0:
        sequential = True
        print(
            f"Detected 'pipe-parallel-size' of {pipeline_world_size}, assuming model is saved as Sequential..."
        )
    else:
        sequential = False
        print(
            f"Detected 'pipe-parallel-size' of {pipeline_world_size}, assuming model is saved as PipelineModule..."
        )

    model = convert(
        args.input_dir,
        loaded_config,
        args.output_dir,
        sequential=sequential,
        precision=args.precision,
    )

    model.save_pretrained(args.output_dir)


if __name__ == "__main__":

    main()

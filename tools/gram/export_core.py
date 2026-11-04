#!/usr/bin/env python3
"""Export the model portion of a TP=1 native NeoX GRAM rank checkpoint.

Output is a weights-only rank checkpoint, for inference or --finetune; it cannot
resume the original optimizer. Only load trusted torch checkpoint files.
"""
import argparse
import json
import importlib.util
from pathlib import Path
import torch

SOURCE = Path(__file__).resolve().parents[2] / "megatron/gram.py"
spec = importlib.util.spec_from_file_location("neox_gram", SOURCE)
gram = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gram)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--level", type=int, choices=[0, 1, 2], default=0,
                        help="0=ordinary core, 1=core+outer, 2=all nested modules")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    if "module" not in checkpoint:
        parser.error("expected a native NeoX/DeepSpeed rank checkpoint containing module")
    if not any(".gram_aux." in k for k in checkpoint["module"]):
        parser.error("input contains no GRAM auxiliary parameters")
    try:
        core = gram.prefix_state_dict(checkpoint["module"], args.level)
    except ValueError as error:
        parser.error(str(error))
    # Drop optimizer/RNG/sampler state deliberately: use only for inference or finetune.
    result = {"module": core, "iteration": 0}
    # Physical prefix exports may have a narrowed core and bias-free adapter
    # outputs. Persist explicit geometry rather than silently rebuilding 4*h.
    geometry = {}
    for name, value in core.items():
        if name.endswith((".mlp.linear1.weight", ".mlp.core.linear1.weight")):
            geometry["intermediate_size"] = value.shape[0]
            geometry["expansion_factor"] = None
            geometry["gram_absorb_auxiliary_width"] = False
            break
    if args.level:
        geometry["gram_aux_output_bias"] = any(".gram_aux.2.bias" in k for k in core)
    geometry.update(gram_enabled=args.level > 0, gram_nested=args.level == 2,
                    gram_eval_level=args.level)
    result["configuration_overrides"] = geometry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    args.output.with_suffix(args.output.suffix + ".config.json").write_text(json.dumps(geometry, indent=2) + "\n")
    print(f"Exported {len(core)} level-{args.level} tensors to {args.output}")


if __name__ == "__main__":
    main()

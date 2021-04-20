# Adapted from https://github.com/awaelchli/pytorch-lightning-snippets/blob/master/checkpoint/peek.py

import code
from argparse import ArgumentParser, Namespace
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch


class COLORS:
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    END = "\033[0m"


PRIMITIVE_TYPES = (int, float, bool, str, type)


def pretty_print(contents: dict):
    """ Prints a nice summary of the top-level contens in a checkpoint dictionary. """
    col_size = max(len(str(k)) for k in contents)
    for k, v in sorted(contents.items()):
        key_length = len(str(k))
        line = " " * (col_size - key_length)
        line += f"{k}: {COLORS.BLUE}{type(v).__name__}{COLORS.END}"
        if isinstance(v, dict):
            pretty_print(v)
        if isinstance(v, PRIMITIVE_TYPES):
            line += f" = "
            line += f"{COLORS.CYAN}{repr(v)}{COLORS.END}"
        elif isinstance(v, Sequence):
            line += ", "
            line += f"{COLORS.CYAN}len={len(v)}{COLORS.END}"
        elif isinstance(v, torch.Tensor):
            if v.ndimension() in (0, 1) and v.numel() == 1:
                line += f" = "
                line += f"{COLORS.CYAN}{v.item()}{COLORS.END}"
            else:
                line += ", "
                line += f"{COLORS.CYAN}shape={list(v.shape)}{COLORS.END}"
                line += ", "
                line += f"{COLORS.CYAN}dtype={v.dtype}{COLORS.END}"
        print(line)


def get_attribute(obj: object, name: str) -> object:
    if isinstance(obj, Mapping):
        return obj[name]
    if isinstance(obj, Namespace):
        return obj.name
    return getattr(object, name)


def peek(args: Namespace):
    files = list(Path(args.dir).glob("*.pt")) + list(Path(args.dir).glob("*.ckpt"))
    for file in files:
        file = Path(file).absolute()
        print(f"{COLORS.GREEN}{file.name}:{COLORS.END}")
        ckpt = torch.load(file, map_location=torch.device("cpu"))
        selection = dict()
        attribute_names = args.attributes or list(ckpt.keys())
        for name in attribute_names:
            parts = name.split("/")
            current = ckpt
            for part in parts:
                current = get_attribute(current, part)
            selection.update({name: current})
        pretty_print(selection)
        print('\n')

        if args.interactive:
            code.interact(
                banner="Entering interactive shell. You can access the checkpoint contents through the local variable 'checkpoint'.",
                local={"checkpoint": ckpt, "torch": torch},
            )


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "dir",
        type=str,
        help="The checkpoint dir to inspect. Must be a directory containing pickle binaries saved with 'torch.save' ending in .pt or .ckpt.",
    )
    parser.add_argument(
        "--attributes",
        nargs="*",
        help="Name of one or several attributes to query. To access an attribute within a nested structure, use '/' as separator.",
        default=None
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Drops into interactive shell after printing the summary.",
    )
    args = parser.parse_args()
    peek(args)


if __name__ == "__main__":
    main()

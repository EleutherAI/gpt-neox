# Copyright (c) 2021, EleutherAI
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
#
# Adapted from https://github.com/awaelchli/pytorch-lightning-snippets/blob/master/checkpoint/peek.py

import code
import os
import re
from argparse import ArgumentParser, Namespace
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch


class COLORS:
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    WHITE = "\033[37m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


PRIMITIVE_TYPES = (int, float, bool, str, type)


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", str(key))]
    return sorted(l, key=alphanum_key)


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


def pretty_print(contents: dict):
    """Prints a nice summary of the top-level contents in a checkpoint dictionary."""
    col_size = max(len(str(k)) for k in contents)
    for k, v in sorted(contents.items()):
        key_length = len(str(k))
        line = " " * (col_size - key_length)
        line += f"{k}: {COLORS.BLUE}{type(v).__name__}{COLORS.END}"
        if isinstance(v, dict):
            pretty_print(v)
        elif isinstance(v, PRIMITIVE_TYPES):
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
            line += (
                ", "
                + f"{COLORS.CYAN}size={sizeof_fmt(v.nelement() * v.element_size())}{COLORS.END}"
            )
        print(line)


def common_entries(*dcts):
    if not dcts:
        return
    for i in set(dcts[0]).intersection(*dcts[1:]):
        yield (i,) + tuple(d[i] for d in dcts)


def pretty_print_double(contents1: dict, contents2: dict, args):
    """Prints a nice summary of the top-level contents in a checkpoint dictionary."""
    col_size = max(
        max(len(str(k)) for k in contents1), max(len(str(k)) for k in contents2)
    )
    common_keys = list(contents1.keys() & contents2.keys())
    uncommon_keys_1 = [i for i in contents2.keys() if i not in common_keys]
    uncommon_keys_2 = [i for i in contents1.keys() if i not in common_keys]
    diffs_found = False
    if uncommon_keys_1 + uncommon_keys_2:
        diffs_found = True
        if uncommon_keys_1:
            print(
                f"{COLORS.RED}{len(uncommon_keys_1)} key(s) found in ckpt 1 that isn't present in ckpt 2:{COLORS.END} \n\t{COLORS.BLUE}{' '.join(uncommon_keys_1)}{COLORS.END}"
            )
        if uncommon_keys_2:
            print(
                f"{COLORS.RED}{len(uncommon_keys_2)} key(s) found in ckpt 2 that isn't present in ckpt 1:{COLORS.END} \n\t{COLORS.BLUE}{' '.join(uncommon_keys_2)}{COLORS.END}"
            )
    for k, v1, v2 in sorted(common_entries(contents1, contents2)):
        key_length = len(str(k))
        line = " " * (col_size - key_length)
        if type(v1) != type(v2):
            print(
                f"{COLORS.RED}{k} is a different type between ckpt1 and ckpt2: ({type(v1).__name__} vs. {type(v2).__name__}){COLORS.END}"
            )
            continue
        else:
            prefix = f"{k}: {COLORS.BLUE}{type(v1).__name__} | {type(v2).__name__}{COLORS.END}"
        if isinstance(v1, dict):
            pretty_print_double(v1, v2, args)
        elif isinstance(v1, PRIMITIVE_TYPES):
            if repr(v1) != repr(v2):
                c = COLORS.RED
                line += f" = "
                line += f"{c}{repr(v1)} | {repr(v2)}{COLORS.END}"
            else:
                c = COLORS.CYAN
                if not args.diff:
                    line += f" = "
                    line += f"{c}{repr(v1)} | {repr(v2)}{COLORS.END}"
        elif isinstance(v1, Sequence):
            if len(v1) != len(v2):
                c = COLORS.RED
                line += ", "
                line += f"{c}len={len(v1)} | len={len(v2)}{COLORS.END}"
            else:
                c = COLORS.CYAN
                if not args.diff:
                    line += ", "
                    line += f"{c}len={len(v1)} | len={len(v2)}{COLORS.END}"
        elif isinstance(v1, torch.Tensor):
            if v1.ndimension() != v2.ndimension():
                c = COLORS.RED
            else:
                c = COLORS.CYAN

            if (v1.ndimension() in (0, 1) and v1.numel() == 1) and (
                v2.ndimension() in (0, 1) and v2.numel() == 1
            ):
                if not args.diff:
                    line += f" = "
                    line += f"{c}{v1.item()} | {c}{v2.item()}{COLORS.END}"
            else:
                if list(v1.shape) != list(v2.shape):
                    c = COLORS.RED
                    line += ", "
                    line += f"{c}shape={list(v1.shape)} | shape={list(v2.shape)}{COLORS.END}"
                else:
                    c = COLORS.CYAN
                    if not args.diff:
                        line += ", "
                        line += f"{c}shape={list(v1.shape)} | shape={list(v2.shape)}{COLORS.END}"
                if v1.dtype != v2.dtype:
                    c = COLORS.RED
                    line += f"{c}dtype={v1.dtype} | dtype={v2.dtype}{COLORS.END}"

                else:
                    c = COLORS.CYAN
                    if not args.diff:
                        line += ", "
                        line += f"{c}dtype={v1.dtype} | dtype={v2.dtype}{COLORS.END}"
                if list(v1.shape) == list(v2.shape):
                    if torch.allclose(v1, v2):
                        if not args.diff:
                            line += f", {COLORS.CYAN}VALUES EQUAL{COLORS.END}"
                    else:
                        line += f", {COLORS.RED}VALUES DIFFER{COLORS.END}"

        if line.replace(" ", "") != "":
            line = prefix + line
            print(line)
            diffs_found = True
    if args.diff and not diffs_found:
        pass
    else:
        if not args.diff:
            print("\n")

    return diffs_found


def get_attribute(obj: object, name: str) -> object:
    if isinstance(obj, Mapping):
        return obj[name]
    if isinstance(obj, Namespace):
        return obj.name
    return getattr(object, name)


def get_files(pth):
    if os.path.isdir(pth):
        files = list(Path(pth).glob("*.pt")) + list(Path(pth).glob("*.ckpt"))
    elif os.path.isfile(pth):
        assert pth.endswith(".pt") or pth.endswith(".ckpt")
        files = [Path(pth)]
    else:
        raise ValueError("Dir / File not found.")
    return natural_sort(files)


def peek(args: Namespace):

    files = get_files(args.dir)

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
        print("\n")

        if args.interactive:
            code.interact(
                banner="Entering interactive shell. You can access the checkpoint contents through the local variable 'checkpoint'.",
                local={"checkpoint": ckpt, "torch": torch},
            )


def get_shared_fnames(files_1, files_2):
    names_1 = [Path(i).name for i in files_1]
    names_1_parent = Path(files_1[0]).parent
    names_2 = [Path(i).name for i in files_2]
    names_2_parent = Path(files_2[0]).parent
    shared_names = list(set.intersection(*map(set, [names_1, names_2])))
    return [names_1_parent / i for i in shared_names], [
        names_2_parent / i for i in shared_names
    ]


def get_selection(filename, args):
    ckpt = torch.load(filename, map_location=torch.device("cpu"))
    selection = dict()
    attribute_names = args.attributes or list(ckpt.keys())
    for name in attribute_names:
        parts = name.split("/")
        current = ckpt
        for part in parts:
            current = get_attribute(current, part)
        selection.update({name: current})
    return selection


def compare(args: Namespace):
    dirs = [i.strip() for i in args.dir.split(",")]
    assert len(dirs) == 2, "Only works with 2 directories / files"
    files_1 = get_files(dirs[0])
    files_2 = get_files(dirs[1])
    files_1, files_2 = get_shared_fnames(files_1, files_2)

    for file1, file2 in zip(files_1, files_2):
        file1 = Path(file1).absolute()
        file2 = Path(file2).absolute()
        print(f"COMPARING {COLORS.GREEN}{file1.name} & {file2.name}:{COLORS.END}")
        selection_1 = get_selection(file1, args)
        selection_2 = get_selection(file2, args)
        diffs_found = pretty_print_double(selection_1, selection_2, args)
        if args.diff and diffs_found:
            print(
                f"{COLORS.RED}THE ABOVE DIFFS WERE FOUND IN {file1.name} & {file2.name} ^{COLORS.END}\n"
            )

        if args.interactive:
            code.interact(
                banner="Entering interactive shell. You can access the checkpoint contents through the local variable 'selection_1' / 'selection_2'.\nPress Ctrl-D to exit.",
                local={
                    "selection_1": selection_1,
                    "selection_2": selection_2,
                    "torch": torch,
                },
            )


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "dir",
        type=str,
        help="The checkpoint dir to inspect. Must be either: \
         - a directory containing pickle binaries saved with 'torch.save' ending in .pt or .ckpt \
         - a single path to a .pt or .ckpt file \
         - two comma separated directories - in which case the script will *compare* the two checkpoints",
    )
    parser.add_argument(
        "--attributes",
        nargs="*",
        help="Name of one or several attributes to query. To access an attribute within a nested structure, use '/' as separator.",
        default=None,
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Drops into interactive shell after printing the summary.",
    )
    parser.add_argument(
        "--compare",
        "-c",
        action="store_true",
        help="If true, script will compare two directories separated by commas",
    )
    parser.add_argument(
        "--diff", "-d", action="store_true", help="In compare mode, only print diffs"
    )

    args = parser.parse_args()
    if args.compare:
        compare(args)
    else:
        peek(args)


if __name__ == "__main__":
    main()

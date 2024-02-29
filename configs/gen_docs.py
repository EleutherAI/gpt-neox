import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from megatron.neox_arguments import neox_args, deepspeed_args
from inspect import getmembers, getsource
from dataclasses import field, is_dataclass
from itertools import tee, zip_longest
import pathlib


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip_longest(a, b)


def get_docs(module):
    ARGS_CLASSES = getmembers(module, is_dataclass)
    results = {}
    for name, dcls in ARGS_CLASSES:
        assert is_dataclass(dcls)
        src = getsource(dcls)
        d = dcls()
        loc = 0
        results[name] = {"doc": d.__doc__.strip(), "attributes": {}}
        for cur, _next in pairwise(d.__dataclass_fields__.items()):
            field_name, field_def = cur
            field_type = field_def.type
            if hasattr(field_type, "__name__"):
                if field_type.__name__ == "Literal" or field_type.__name__ == "Union":
                    field_type = field_type
                else:
                    field_type = str(field_type.__name__)
            else:
                field_type = str(field_type)

            field_default = field_def.default

            # try to find the field definition
            loc = src.find(f" {field_name}:", loc + len(field_name) + 1)

            if _next is not None:
                next_field_name, _ = _next
                # try to find the next field definition
                next_loc = src.find(f"{next_field_name}:", loc + len(field_name))
            else:
                next_loc = len(src)

            # try to get the docstring
            _src = src[loc:next_loc].strip()
            if '"""' in _src:
                doc = _src.split('"""')[1].strip()
            elif "'''" in _src:
                doc = _src.split("'''")[1].strip()
            else:
                doc = ""
            results[name]["attributes"][field_name] = {
                "name": field_name,
                "type": field_type,
                "default": field_default,
                "doc": doc,
            }
    return results


def to_md(docs, intro_str=""):
    """
    Writes the docs dictionary to markdown format
    """
    lines = []
    lines.append(intro_str)
    for name, doc in docs.items():
        lines.append(f"## {name}")
        lines.append(f"{doc['doc']}")
        lines.append("")
        for field_name, field_def in doc["attributes"].items():
            # attribute name and type
            lines.append(f"- **{field_name}**: {field_def['type']}")
            # default value
            lines.append(f"    Default = {str(field_def['default'])}")
            lines.append(f"    {field_def['doc']}")
            lines.append("")
    return "\n\n".join(lines)


if __name__ == "__main__":
    docs = get_docs(neox_args)
    docs.update(get_docs(deepspeed_args))
    intro_str = """Arguments for gpt-neox. All of the following can be specified in your .yml config file(s):\n"""
    md = to_md(docs, intro_str=intro_str)
    with open(f"{pathlib.Path(__file__).parent.resolve()}/neox_arguments.md", "w") as f:
        f.write(md)

# Copyright (c) 2025, EleutherAI
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

"""
load all confings in neox/configs in order to perform validations implemented in NeoXArgs
"""
from unittest.mock import Mock

import pytest
import yaml
from ..common import get_configs_with_path


def run_neox_args_load_test(yaml_files):
    from megatron.neox_arguments import NeoXArgs

    yaml_list = get_configs_with_path(yaml_files)
    args_loaded = NeoXArgs.from_ymls(yaml_list)
    assert isinstance(args_loaded, NeoXArgs)

    # initialize an empty config dictionary to be filled by yamls
    config = dict()

    # iterate of all to be loaded yaml files
    for conf_file_name in yaml_list:

        # load file
        with open(conf_file_name) as conf_file:
            conf = yaml.load(conf_file, Loader=yaml.FullLoader)

        # check for key duplicates and load values
        for conf_key, conf_value in conf.items():
            if conf_key in config:
                raise ValueError(
                    f"Conf file {conf_file_name} has the following duplicate keys with previously loaded file: {conf_key}"
                )

            conf_key_converted = conf_key.replace(
                "-", "_"
            )  # TODO remove replace and update configuration files?
            config[conf_key_converted] = conf_value

    # validate that neox args has the same value as specified in the config (if specified in the config)
    for k, v in config.items():
        neox_args_value = getattr(args_loaded, k)
        assert v == neox_args_value, (
            "loaded neox args value "
            + str(k)
            + " == "
            + str(neox_args_value)
            + " different from config file "
            + str(v)
        )


@pytest.mark.cpu
def test_gradient_clipping_is_passed_to_deepspeed():
    """An explicitly configured value must not be dropped as a NeoX default."""
    from megatron.neox_arguments import NeoXArgs

    args_loaded = NeoXArgs.from_ymls(
        get_configs_with_path(["125M.yml", "local_setup.yml", "cpu_mock_config.yml"])
    )

    assert args_loaded.gradient_clipping == 1.0
    assert args_loaded.deepspeed_config["gradient_clipping"] == 1.0


@pytest.mark.cpu
@pytest.mark.parametrize(
    "name,value",
    [
        ("minimum_tokens", -1),
        ("maximum_tokens", -1),
        ("temperature", -0.1),
        ("top_k", -1),
        ("top_p", 1.01),
        ("min_p", -0.01),
        ("top_h", 1.01),
        ("typical_p", 0.0),
        ("typical_p", 1.01),
        ("epsilon_cutoff", 1.0),
        ("eta_cutoff", 1.0),
        ("eta_cutoff", float("nan")),
    ],
)
def test_config_rejects_invalid_sampling_before_model_setup(name, value):
    from megatron.neox_arguments import NeoXArgs

    with pytest.raises(ValueError, match=name):
        NeoXArgs.from_ymls(
            get_configs_with_path(
                ["125M.yml", "local_setup.yml", "cpu_mock_config.yml"]
            ),
            overwrite_values={"log_dir": None, name: value},
        )


@pytest.mark.cpu
def test_config_rejects_minimum_above_maximum():
    from megatron.neox_arguments import NeoXArgs

    with pytest.raises(ValueError, match=r"minimum_tokens \(5\).*maximum_tokens \(4\)"):
        NeoXArgs.from_ymls(
            get_configs_with_path(
                ["125M.yml", "local_setup.yml", "cpu_mock_config.yml"]
            ),
            overwrite_values={
                "log_dir": None,
                "minimum_tokens": 5,
                "maximum_tokens": 4,
            },
        )


@pytest.mark.cpu
def test_sampling_boundaries_and_runtime_overrides(monkeypatch):
    from megatron.neox_arguments import NeoXArgs
    from megatron.neox_arguments.sampling import SAMPLING_RANGES
    from megatron.text_generation_utils import stream_tokens

    options = dict(
        minimum_tokens=0,
        maximum_tokens=0,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        min_p=0.0,
        top_h=1.0,
        typical_p=1.0,
        epsilon_cutoff=0.0,
        eta_cutoff=0.999,
    )
    with pytest.warns(
        UserWarning, match="will be ignored: top_h=1.0, eta_cutoff=0.999"
    ):
        args = NeoXArgs.from_ymls(
            get_configs_with_path(
                ["125M.yml", "local_setup.yml", "cpu_mock_config.yml"]
            ),
            overwrite_values={"log_dir": None, **options},
        )
    # No range predicate should run again for unchanged options. A mutated
    # config or a request override must still fail before touching the model.
    for name, (accepts, domain) in SAMPLING_RANGES.items():
        monkeypatch.setitem(SAMPLING_RANGES, name, (Mock(wraps=accepts), domain))
    model = Mock()
    model.eval.side_effect = RuntimeError("reached model")
    with pytest.raises(RuntimeError, match="reached model"):
        next(stream_tokens(args, model, [[1]], **options))
    assert all(check.call_count == 0 for check, _ in SAMPLING_RANGES.values())

    args.top_h = options["top_h"] = -0.1
    with pytest.raises(ValueError, match="top_h"):
        next(stream_tokens(args, model, [[1]], **options))
    assert SAMPLING_RANGES["top_h"][0].call_count == 1
    assert model.eval.call_count == 1


@pytest.mark.cpu
def test_clip_grad_with_gradient_clipping_raises():
    from megatron.neox_arguments import NeoXArgs

    with pytest.raises(ValueError, match="sets both 'clip_grad: 0.5'"):
        NeoXArgs.from_ymls(
            get_configs_with_path(
                ["125M.yml", "local_setup.yml", "cpu_mock_config.yml"]
            ),
            overwrite_values={"clip_grad": 0.5},
        )


@pytest.mark.cpu
@pytest.mark.parametrize("attention_type", ["gmlp", "amlp"])
def test_mup_rejects_gmlp_attention(attention_type):
    from megatron.neox_arguments import NeoXArgs

    with pytest.raises(ValueError, match="use_mup does not support gmlp or amlp"):
        NeoXArgs.from_ymls(
            get_configs_with_path(
                ["125M.yml", "local_setup.yml", "cpu_mock_config.yml"]
            ),
            overwrite_values={
                "use_mup": True,
                "partition_activations": False,
                "attention_config": [[[attention_type], "all"]],
            },
        )


@pytest.mark.cpu
def test_strip_deprecated_config_keys(monkeypatch):
    from megatron.neox_arguments import NeoXArgs, arguments

    strip = NeoXArgs._strip_deprecated_config_keys
    assert strip({"clip_grad": 0.5}) == {"gradient_clipping": 0.5}

    monkeypatch.setitem(arguments.DEPRECATED_ARGS, "removed_option", None)
    with pytest.raises(ValueError, match="'removed_option' has been removed"):
        strip({"removed_option": 1})
    with pytest.raises(ValueError, match="'sparse_attention' has been removed"):
        strip({"sparse_attention": {"block": 32}})


@pytest.mark.cpu
def test_gradient_clipping_default_matches_deepspeed():
    from deepspeed.runtime.constants import GRADIENT_CLIPPING_DEFAULT
    from megatron.neox_arguments.deepspeed_args import NeoXArgsDeepspeedConfig

    assert NeoXArgsDeepspeedConfig().gradient_clipping == GRADIENT_CLIPPING_DEFAULT


@pytest.mark.cpu
def test_neoxargs_load_arguments_125M_local_setup():
    """
    verify 125M.yml can be loaded without raising validation errors
    """
    run_neox_args_load_test(["125M.yml", "local_setup.yml", "cpu_mock_config.yml"])


@pytest.mark.cpu
def test_neoxargs_load_arguments_125M_local_setup_text_generation():
    """
    verify 125M.yml can be loaded together with text generation without raising validation errors
    """
    run_neox_args_load_test(
        ["125M.yml", "local_setup.yml", "text_generation.yml", "cpu_mock_config.yml"]
    )


@pytest.mark.cpu
def test_neoxargs_load_arguments_350M_local_setup():
    """
    verify 350M.yml can be loaded without raising validation errors
    """
    run_neox_args_load_test(["350M.yml", "local_setup.yml", "cpu_mock_config.yml"])


@pytest.mark.cpu
def test_neoxargs_load_arguments_760M_local_setup():
    """
    verify 760M.yml can be loaded without raising validation errors
    """
    run_neox_args_load_test(["760M.yml", "local_setup.yml", "cpu_mock_config.yml"])


@pytest.mark.cpu
def test_neoxargs_load_arguments_2_7B_local_setup():
    """
    verify 2-7B.yml can be loaded without raising validation errors
    """
    run_neox_args_load_test(["2-7B.yml", "local_setup.yml", "cpu_mock_config.yml"])


@pytest.mark.cpu
def test_neoxargs_load_arguments_6_7B_local_setup():
    """
    verify 6-7B.yml can be loaded without raising validation errors
    """
    run_neox_args_load_test(["6-7B.yml", "local_setup.yml", "cpu_mock_config.yml"])


@pytest.mark.cpu
def test_neoxargs_load_arguments_13B_local_setup():
    """
    verify 13B.yml can be loaded without raising validation errors
    """
    run_neox_args_load_test(["13B.yml", "local_setup.yml", "cpu_mock_config.yml"])


@pytest.mark.cpu
def test_neoxargs_load_arguments_1_3B_local_setup():
    """
    verify 1-3B.yml can be loaded without raising validation errors
    """
    run_neox_args_load_test(["1-3B.yml", "local_setup.yml", "cpu_mock_config.yml"])


@pytest.mark.cpu
def test_neoxargs_load_arguments_175B_local_setup():
    """
    verify 13B.yml can be loaded without raising validation errors
    """
    run_neox_args_load_test(["175B.yml", "local_setup.yml", "cpu_mock_config.yml"])


@pytest.mark.cpu
def test_neoxargs_fail_instantiate_without_required_params():
    """
    verify assertion error if required arguments are not provided
    """

    try:
        run_neox_args_load_test(["local_setup.yml"])
        assert False
    except Exception as e:
        assert True


@pytest.mark.cpu
def test_neoxargs_fail_instantiate_without_any_params():
    """
    verify assertion error if required arguments are not provided
    """
    from megatron.neox_arguments import NeoXArgs

    try:
        args_loaded = NeoXArgs()
        assert False
    except Exception as e:
        assert True

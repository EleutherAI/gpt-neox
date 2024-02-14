# Copyright (c) 2024, EleutherAI
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

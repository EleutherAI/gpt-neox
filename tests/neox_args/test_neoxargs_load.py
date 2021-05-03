from megatron.neox_arguments import NeoXArgs

from ..common import get_configs_with_path

def test_neoxargs_load_arguments_small_local_setup():
    """
    verify small.yml can be loaded without raising validation errors
    """
    yaml_list = get_configs_with_path(["small.yml", "local_setup.yml"])
    args_loaded = NeoXArgs.from_ymls(yaml_list)
    assert isinstance(args_loaded, NeoXArgs)

def test_neoxargs_load_arguments_small_local_setup_text_generation():
    """
    verify small.yml can be loaded together with text generation without raising validation errors
    """
    yaml_list = get_configs_with_path(["small.yml", "local_setup.yml", "text_generation.yml"])
    args_loaded = NeoXArgs.from_ymls(yaml_list)
    assert isinstance(args_loaded, NeoXArgs)
    
def test_neoxargs_load_arguments_medium_local_setup():
    """
    verify medium.yml can be loaded without raising validation errors
    """
    yaml_list = get_configs_with_path(["medium.yml", "local_setup.yml"])
    args_loaded = NeoXArgs.from_ymls(yaml_list)
    assert isinstance(args_loaded, NeoXArgs)

def test_neoxargs_load_arguments_large_local_setup():
    """
    verify large.yml can be loaded without raising validation errors
    """
    yaml_list = get_configs_with_path(["large.yml", "local_setup.yml"])
    args_loaded = NeoXArgs.from_ymls(yaml_list)
    assert isinstance(args_loaded, NeoXArgs)

def test_neoxargs_load_arguments_2_7B_local_setup():
    """
    verify 2-7B.yml can be loaded without raising validation errors
    """
    yaml_list = get_configs_with_path(["2-7B.yml", "local_setup.yml"])
    args_loaded = NeoXArgs.from_ymls(yaml_list)
    assert isinstance(args_loaded, NeoXArgs)

def test_neoxargs_load_arguments_6_7B_local_setup():
    """
    verify 6-7B.yml can be loaded without raising validation errors
    """
    yaml_list = get_configs_with_path(["6-7B.yml", "local_setup.yml"])
    args_loaded = NeoXArgs.from_ymls(yaml_list)
    assert isinstance(args_loaded, NeoXArgs)

def test_neoxargs_load_arguments_13B_local_setup():
    """
    verify 13B.yml can be loaded without raising validation errors
    """
    yaml_list = get_configs_with_path(["13B.yml", "local_setup.yml"])
    args_loaded = NeoXArgs.from_ymls(yaml_list)
    assert isinstance(args_loaded, NeoXArgs)

def test_neoxargs_load_arguments_XL_local_setup():
    """
    verify XL.yml can be loaded without raising validation errors
    """
    yaml_list = get_configs_with_path(["XL.yml", "local_setup.yml"])
    args_loaded = NeoXArgs.from_ymls(yaml_list)
    assert isinstance(args_loaded, NeoXArgs)

def test_neoxargs_load_arguments_175B_local_setup():
    """
    verify 13B.yml can be loaded without raising validation errors
    """
    yaml_list = get_configs_with_path(["175B.yml", "local_setup.yml"])
    args_loaded = NeoXArgs.from_ymls(yaml_list)
    assert isinstance(args_loaded, NeoXArgs)

def test_neoxargs_fail_instantiate_without_required_params():
    """
    verify assertion error if required arguments are not provided
    """
    try:
        yaml_list = get_configs_with_path(["local_setup.yml"])
        args_loaded = NeoXArgs.from_ymls(yaml_list)
        assert False
    except Exception as e:
        assert True

def test_neoxargs_fail_instantiate_without_any_params():
    """
    verify assertion error if required arguments are not provided
    """
    try:
        args_loaded = NeoXArgs()
        assert False
    except Exception as e:
        assert True


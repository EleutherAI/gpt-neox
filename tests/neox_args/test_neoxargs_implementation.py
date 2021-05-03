from megatron import NeoXArgs

def test_neoxargs_duplicates():
    """
    tests that there are no duplicates among parent classes of NeoXArgs
    """
    assert NeoXArgs.validate_keys(), "test_neoxargs_duplicates"
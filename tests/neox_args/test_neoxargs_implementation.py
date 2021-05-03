"""
check implementation of NeoXArgs for duplication errors (would overwrite)
"""

def test_neoxargs_duplicates():
    """
    tests that there are no duplicates among parent classes of NeoXArgs
    """
    from megatron import NeoXArgs
    assert NeoXArgs.validate_keys(), "test_neoxargs_duplicates"
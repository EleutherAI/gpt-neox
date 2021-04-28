"""
run with 

```python
pytest run_tests.py
```
"""


import unittest

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover('tests')

    unittest.TextTestRunner(failfast=True).run(suite)
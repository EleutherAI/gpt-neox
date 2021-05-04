"""
run with 

```python
pytest run_tests.py
```
"""

import unittest
from tests import *
import logging
from pathlib import Path

log_file = Path("logs") / "test.log"
log_file.parent.mkdir(exist_ok=True, parents=True)
logging.basicConfig(format='%(levelname)s: %(message)s in %(pathname)s line %(lineno)d', level=logging.INFO, filename=str(log_file))

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover('tests')

    unittest.TextTestRunner(failfast=True).run(suite)

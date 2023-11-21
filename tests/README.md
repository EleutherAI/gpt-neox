# Dependencies

Tests use pytests with coverage and forked plugins. Install with:

```bash
pip install -r requirements/requirements-dev.txt
```

Download the required test data
```bash
python prepare_data.py
```

# Run

Tests can be run using pytest.

* The argument --forked needs to be provided
* A coverage report can be created using the optional arguments --cov-report and --cov (see pytest documentation)
* A subset of tests can be selected by pointing to the module within tests

```bash
# run all tests, output coverage report of megatron module in terminal
pytest --forked --cov-report term --cov=megatron tests

# run tests in tests/model, output coverage report of megatron module as html
pytest --forked --cov-report html --cov=megatron tests/model

# run tests in tests/model/test_model_generation.py, don't output coverage report
pytest --forked tests/model/test_model_generation.py
```

Some tests can run on cpu only. These are marked with the decorator @pytest.mark.cpu.
The test cases for cpu can be run with:
````
pytest tests -m cpu
```

If a html coverage report has been created a simple http server can be run to serve static files.

```bash
python -m http.server --directory htmlcov 8000
```


## Tips and Tricks
if You see this kind of error:
```
RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
```
It usually means that you used some pytorch.cuda function before the test creates the processes. However just importing `from torch.utils import cpp_extension` can also trigger this.

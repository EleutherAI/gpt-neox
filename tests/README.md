# Dependencies

Tests use pytests with coverage and html output plugins. Install with:

```bash
pip install -r requirements/requirements-dev.txt
```

Download the required test data
```bash
python prepare_data.py
```

# Run

Tests can be run using pytest.

* A coverage report can be created using the optional arguments --cov-report and --cov (see pytest documentation)
* A subset of tests can be selected by pointing to the module within tests

```bash
# run all tests, output coverage and html report of megatron module in terminal
pytest --cov-report term --cov=megatron --html=test-report.html --self-contained-html tests

# run tests in tests/model, output coverage report and test report of megatron module as html
pytest --cov-report html --cov=megatron --html=model-test--report.html --self-contained-html tests/model

# run tests in tests/model/test_model_generation.py, don't output coverage report
pytest tests/model/test_model_generation.py
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

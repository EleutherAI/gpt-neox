# Dependencies

Tests use pytests with coverage and forked plugins. Install with:

```bash
pip install -r requirements/requirements-dev.txt
```

# Run

Tests can be run using pytest. 

* The argument --forked needs to be provided
* A coverage report can be created using the optional arguments --cov-report and --cov (see pytest documentation)

```bash
# run all tests, output coverage report of megatron module in terminal
pytest --forked --cov-report term --cov=megatron tests

# run tests in tests/model, output coverage report of megatron module as html
pytest --forked --cov-report html --cov=megatron tests/model
```

If a html coverage report has been created a simple http server can be run to serve static files.

```bash
python -m http.server --directory htmlcov 8000
```

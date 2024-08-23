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
```
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


## CPU Test Integration

Tests can be run against physical CPUs through GitHub Actions. To have tests run on the physical CPU test, here is generally how the CI should be written:

### runs-on

#### NOTE: These BKMs were written to work with CI infrastructure that is no longer in place. To use the Github runners (ubuntu-22.04 / ubuntu-latest), skip the 'runs-on' section.

The CI needs to be written to target the CPU Github Action runner. The jobs that need to run on CPU should use the hardware runner's labels:
```yaml
jobs:
  cpu-test-job:
    runs-on: [ 'self-hosted', 'aws', 'test'] # these labels tell GitHub to execute on the runner with the 'aws' and 'test' labels
```

### Software dependencies

Hardware tests that need python and docker should install them as part of the test execution to make sure the tests run as expected:
```yaml
steps:
    # sample syntax to setup python with pip
  - uses: actions/setup-python@v4
    with:
      python-version: "3.8"
      cache: "pip"

    # sample setup of docker (there's no official Docker setup action)
  - name: Docker setup
    run: | # taken from Docker's installation page: https://docs.docker.com/engine/install/ubuntu/
      # Add Docker's official GPG key:
      sudo apt-get update
      sudo apt-get install ca-certificates curl
      sudo install -m 0755 -d /etc/apt/keyrings
      sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
      sudo chmod a+r /etc/apt/keyrings/docker.asc
      # Add the repository to Apt sources:
      echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
        $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
        sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
      sudo apt-get update
      sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y
```

Any other software dependencies should be assumed to be missing and installed as part of the CI.

### Using Docker image

Using the Docker image and running tests in a container is recommended to resolve environment issues. There is a modified docker-compose.yml in tests/cpu_tests directory that is recommended to be used for CPU tests:

```bash
cp tests/cpu_tests/docker-compose.yml .
# export any env variables here that should be used:
export NEOX_DATA_PATH='./data/enwik8'
docker compose run -d --build --name $CONTAINER gpt-neox tail -f /dev/null
# then can set up and run tests in the container using docker exec
docker exec $CONTAINER pip install -r /workspace/requirements-dev.txt
# etc.
# please clean up the container as part of the CI:
docker rm $CONTAINER
```

At the time of writing there is no built-in method to provide an offline-built Docker image to `jobs.<job-id>.container`.

### Using existing CPU test CI

There is an existing CPU test workflow that can be included in existing CI:

```yaml
steps:
  - name: Run CPU Tests
    uses:
      target_test_ref: $GITHUB_REF # replace with the ref/SHA that the tests should be run on
      # have a look at the reusable workflow here: https://github.com/EleutherAI/gpt-neox/blob/main/tests/cpu_tests/action.yml
```

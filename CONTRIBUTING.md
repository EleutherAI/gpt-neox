# Contributing
GPT-NeoX welcomes your contributions!

## Prerequisites
GPT-NeoX uses [pre-commit](https://pre-commit.com/) to ensure that formatting is
consistent across GPT-NeoX. First, ensure that `pre-commit` is installed with
`pip install pre-commit`. Next, the pre-commit hooks must be installed once
before commits can be made:
```bash
pre-commit install
```
Please install `clang-format` from Conda:
```bash
conda install clang-format
```

Afterwards, our suite of formatting tests run automatically before each `git commit`. You
can also run these manually:
```bash
pre-commit run --all-files
```
If a formatting test fails, it will fix the modified code in place and abort
the `git commit`. After looking over the changes, you can `git add <modified files>`
and then repeat the previous `git commit` command.


## Testing
GPT-NeoX tracks two types of tests: unit tests and more costly model convergence tests.
Unit tests are found in `tests/unit/` and the model convergence tests are found in
`tests/model/`.

### Unit Tests
[PyTest](https://docs.pytest.org/en/latest/) is used to execute tests. PyTest can be
installed from PyPI via `pip install pytest`. Simply invoke `pytest --forked` to run the
unit tests:
```bash
pytest --forked tests/unit/
```
You can also provide the `-v` flag to `pytest` to see additional information about the
tests. Note that [pytest-forked](https://github.com/pytest-dev/pytest-forked) and the
`--forked` flag are required to test CUDA functionality in distributed tests.

### Model Tests
To execute model tests, first install GPT-NeoX. Next, execute the model test driver:
```bash
cd tests/model/
pytest run_sanity_check.py
```
Note that the `--forked` flag is not necessary for the model tests.

## Contributor License Agreement
This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to, and
actually do, grant us the rights to use your contribution. For details, visit
https://cla-assistant.io/EleutherAI/gpt-neox.

When you submit a pull request, a CLA bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply
follow the instructions provided by the bot. You will only need to do this once across
all repos using our CLA.

## New Feature Contribution Guidelines
Unlike bug fix or improving existing feature (where users usually directly submit a PR and we review it), adding a new feature to GPT-NeoX requires several steps: (1) proposal and discussion, (2) implementation and verification, (3) release and maintenance. This general guideline applies to all new feature contributions. Core GPT-NeoX team member contributions may complete step 1 internally.

### Step 1: Proposal and Discussion
We ask users to first post your intended feature in an issue. This issue needs to include:

* A description of the proposed feature.
* A motivation of why it will be useful to GPT-NeoX users.
* A rough design of how you implement the feature inside GPT-NeoX.
* (Important) Results or planned experiments to demonstrate the effectiveness and correctness of the feature.
  * If the feature only affects performance and does not affect training convergence, we require testing on a fraction of training to demonstrate that the training/validation loss are consistent with baseline, and that the performance is better than baseline.
  * If the feature does affect training convergence, we require testing the whole training to demonstrate that the feature achieves better/on-par final model quality and training performance compared to baseline.

Based on the issue we shall discuss the merit of the new feature and decide whether to accept or decline the proposal. Once accepted and after we confirm the design and implementation plan, we are ready for step 2.

### Step 2: Implementation and Verification
The contributor will proceed and implement the feature, and the GPT-NeoX team will provide guidance/helps as needed. The required deliverables include:

* A PR to [EleutherAI/GPT-NeoX](https://github.com/EleutherAI/gpt-neox) including (1) the feature implementation (2) unit tests (3) documentation (4) example usage.
* In the implementation (code, documentation, tutorial), we require the feature author to record their GitHub username as a contact method for future questions/maintenance.

After receiving the PRs, we will review them and merge them after necessary tests/fixes.

### Step 3: Release and Maintenance
After the PRs are merged, we will announce the feature on our website (with credit to the feature author). We ask the feature author to commit to the maintenance of the feature.

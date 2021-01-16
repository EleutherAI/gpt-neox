## Instructions for running test_dataset.py

1. Install gpt_neox repo (this one)
```
cd gpt_neox
pip install -e .
```
2. Edit the file locations in dataset_test_cases.py to make sure that they point to your correct data paths. Also change any params you'd like
3. Run dataset_test_cases.py
```
python tests/dataset_test_cases.py
```
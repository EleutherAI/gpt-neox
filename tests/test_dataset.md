## Instructions for running test_dataset.py
1. Install the-pile dataset
```
git clone https://github.com/EleutherAI/the-pile.git
```
2. Download all datasets from the pile
```
cd the-pile
pip install -e .
python the_pile/pile.py --force_download
```
3. Install gpt_neox repo (this one)
```
cd gpt_neox
pip install -e .
```
4. Decompress a jsonl file for use in test_dataset.py (by default we use the enron emails dataset)
To decompress enron emails:
```
cd the-pile/components/enron_emails
zstd -d enron_emails.jsonl.zst
```
5. Edit the file locations in test_dataset.py to make sure that they point to your correct data paths. Also check whether you're pre-tokenizing or not
6. Run test_dataset.py
```
cd gpt_neox/tests
python test_dataset.py
```
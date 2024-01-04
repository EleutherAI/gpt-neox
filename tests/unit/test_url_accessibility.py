import pytest
import requests

from tools.datasets.corpora import DATA_DOWNLOADERS


def check_url_accessible(url):
    try:
        response = requests.head(url, timeout=5)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error: Unable to access URL - {e}")
        return False


@pytest.mark.cpu
@pytest.mark.parametrize("dataset_name", list(DATA_DOWNLOADERS.keys()))
def test_url_accessibility(dataset_name):
    if dataset_name == "pass":
        return
    elif not dataset_name == "enwik8":
        pytest.xfail()
    for url in DATA_DOWNLOADERS[dataset_name].urls:
        assert check_url_accessible(url)

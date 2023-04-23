import jsonlines
import datasets
import os, glob
from tqdm import tqdm

batch_size = 1024
NUM_PROC = max(1, os.cpu_count() - 4)


def batch_iterator(
    dataset: datasets.arrow_dataset.Dataset,
    key: str = "text",
    batch_size: int = batch_size,
):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size][key]


def text_iterator(text_path: str, key: str = "text", batch_size: int = batch_size):
    try:
        with open(
            text_path,
            "r",
            buffering=1024,
        ) as input_txt:
            while batch := input_txt.readlines(batch_size):
                yield batch
    except:
        return None


def load_from_filepath(path: str, key="text"):
    dataset = None
    if os.path.isfile(path):
        if ".jsonl" in path:
            dataset = load_dataset(path)
        if ".txt" in path:
            dataset = load_text(path)
    return dataset


def load_from_path(path: str, key="text"):
    path = os.path.normpath(path).replace("~", os.path.expanduser("~"))
    if os.path.isfile(path):
        dataset = load_from_filepath(path)

    if os.path.isdir(path):
        try:
            # attempt to approach as single arrow Dataset dir
            dataset = datasets.load_from_disk(path)  # returns dataset
        except FileNotFoundError:
            ds_list = []
            globpath = os.path.join(path, "*")
            subpaths = glob.glob(globpath)
            for subpath in subpaths:
                if os.path.isfile(subpath):
                    ds = load_from_filepath(subpath)
                    if ds is not None:
                        ds_list.append(ds)
                if os.path.isdir(subpath):
                    try:
                        dataset = datasets.load_from_disk(subpath)
                    except:
                        # don't go recursive.
                        pass
            if ds_list == []:
                dataset = None
            else:
                dataset = datasets.concatenate_datasets(ds_list)

    return dataset


def load_dataset(file_path, key="text"):
    data = []
    with jsonlines.open(file_path) as reader:
        for sample in reader.iter(allow_none=True, skip_empty=True, skip_invalid=True):
            try:
                data.append({key: sample[key]})
            except:
                pass
    dataset = datasets.Dataset.from_list(data)
    return dataset


def load_text(text_path: str, key: str = "text", num_proc: int = NUM_PROC):
    dataset = datasets.Dataset.from_text(text_path, num_proc)
    return dataset


def convert_json_to_dataset(file_path, save_path, key="text"):
    dataset = load_dataset(file_path, key)
    dataset.save_to_disk(save_path)

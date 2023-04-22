import jsonlines
import datasets
import os

print()


def batch_iterator(
    dataset: datasets.arrow_dataset.Dataset,
    key: str = "text",
    batch_size: int = 1024,
):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size][key]


def load_dataset(file_path, key="text"):
    data = []
    file_path = file_path.replace("~", os.path.expanduser("~"))
    with jsonlines.open(file_path) as reader:
        for sample in reader.iter():
            try:
                data.append({key: sample[key]})
            except:
                pass
    dataset = datasets.Dataset.from_list(data)
    return dataset


def convert_json_to_dataset(file_path, save_path, key="text"):
    dataset = load_dataset(file_path, key)
    dataset.save_to_disk(save_path)

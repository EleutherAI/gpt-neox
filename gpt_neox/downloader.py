import os

DATASETS = {
    "owt2": "http://eaidata.bmk.sh/data/owt2_new.tar.gz",
    "enwiki8": "http://eaidata.bmk.sh/data/enwik8.gz"
}


def download_dataset(dataset, dataset_dir="./data"):
    if DATASETS.get(dataset, False):
        return _download_dataset(DATASETS[dataset], os.path.join(dataset_dir, dataset))
    else:
        raise NotImplementedError


def _download_dataset(dataset_url, dataset_dir):
    file_name = os.path.basename(dataset_url)
    output_path = os.path.join(dataset_dir, file_name)

    if not os.path.isfile(output_path):
        os.system('mkdir -p {}'.format(dataset_dir))
        os.system('wget {} -O {}'.format(dataset_url, output_path))

    return output_path

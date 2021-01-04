import os
import tarfile


def download_dataset(dataset, dataset_dir="/root/data"):
    if dataset == "OWT2":
        _download_owt2(dataset_dir)
    else:
        raise NotImplementedError  # TODO: tokenize text data on the fly


def _download_owt2(dataset_dir):
    download_url = "http://eaidata.bmk.sh/data/owt2_new.tar.gz"
    file_name = os.path.basename(download_url)
    output_path = os.path.join(dataset_dir, file_name)

    if not os.path.isfile(output_path):
        os.system('mkdir -p {}'.format(dir))
        os.system('wget  -O {}'.format(output_path))

    dataset_tar = tarfile.open(output_path)
    dataset_tar.extractall(dataset_dir)
    dataset_tar.close()

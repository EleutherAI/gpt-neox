from datasets import load_dataset, get_dataset_config_names

from multiprocess import Pool
def save_config(config):
    load_dataset("bigscience/P3",config).save_to_disk(f'./data/{config}/')
if __name__ == '__main__':
    configs = get_dataset_config_names('bigscience/P3')
    with Pool(80) as p:
        p.map(save_config,configs[481:])  
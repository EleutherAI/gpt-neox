from datasets import get_dataset_config_names, load_dataset, load_from_disk
import datasets
import re
import subprocess
import multiprocessing
import lm_dataformat
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import random
import json
import os
import shutil

DATASETS = {
    'multiple_choice' : [
        'common_sense',
        'dream',
        'quail',
        'quartz',
        'social',
        'wiqa',
        'cosmos',
        'qasc',
        'quarel',
        'sciq',
        'wiki_hop',
        'arc',
        'openbookqa',
        'multirc',
        'piqa',
        'race_high',    
        'boolq'
    ],
    'extractive' : [
        'adversarial',
        'quoref',
        'duorc',
        'ropes',
        'squad',
        'record'
    ],
    'close' : [
        'hotpot',
        'wiki_qa',
        'trivia_qa',
        'web_questions'
    ],
    'sentiment' : [
        'amazon',
        'app_reviews',
        'imdb',
        'rotten_tomatoes',
        'yelp',
    ],
    'summarization' : [
        'cnn_dailymail',
        'gigaword',
        'multi_news',
        'samsum',
        'xsum'
    ],
    'topic_classification' : [
        'ag_news',
        'dbpedia',
        'trec'
    ],
    'paraphase' : [
        'mrpc',
        'paws',
        'qqp'
    ],
    'structure_to_text' : [
        'common_gen',
        'wiki_bio'
    ]
}


def get_dataset(config):
    '''Checks if a path can be included in the train/test/validation dataset'''
    for section in DATASETS.values():
        for pattern in section:
            if(re.search(pattern,config)):
                return pattern
    
    return 'others' # Evaluation dataset

def process_text(text):
    '''Strips the text from any spaces (or) new lines to maintain consistancy'''
    text = text.strip(' \n')
    return text

def process_batch(prompt,response,key,arr):
    '''Processes a single element of P3 dataset'''
    prompt = process_text(prompt)
    response = process_text(response)
    text = f'{prompt}\n{response}'
    arr.append(text)

def process_hf_dataset(ds,key,config):
    '''Process a single split (train/test/validation) of a P3 dataset'''
    
    arr = []
    
    if('is_correct' in ds.column_names):
        ds.filter(lambda x:x['is_correct'])
    ds.map(lambda inputs,targets :process_batch(inputs,targets,key,arr),input_columns=['inputs_pretokenized','targets_pretokenized'])
    
    if(not os.path.exists(f'./jsondata/{key}/')):
        os.mkdir(f'./jsondata/{key}/')
    with open(f'./jsondata/{key}/{config}.json','w') as f:
        json.dump(arr,f)

def process(config,lock):
    '''Driver function to process a single config file of P3 dataset'''
    
    print(f"processing {config}")
    with lock:
        hf_ds = load_from_disk(f'/mnt/ssd-1/P3/hfdataset/{config}')
        print(f"loaded {config}")
    for key in hf_ds.keys():
        if(key != 'train'):
            
            if(key == 'valid'):
                key = 'validation'
            if(key not in ['test','validation']):
                continue
            
            process_hf_dataset(hf_ds[key],key,config)
        else:
            process_hf_dataset(hf_ds[key],get_dataset(config),config)
    
    print(f"processed {config}")


def tokenize(key):
    '''tokenizes the text in config file to a megatron compatible format'''
    
    if(key == 'train'):
        configs = [config for category in DATASETS for config in DATASETS[category]]
        configs = ','.join([f'/mnt/ssd-cluster/P3_configs/{config}' for config in configs])
        path = "/mnt/ssd-cluster/P3_combined/train"
    elif key == 'others':
        configs = f"/mnt/ssd-cluster/P3_configs/{key}"
        path = f"/mnt/ssd-cluster/P3_combined/evaluation"
    else:
        configs = f"/mnt/ssd-cluster/P3_configs/{key}"
        path = f"/mnt/ssd-cluster/P3_combined/{key}"
    
    exec_command = f"cd /home/mchorse/gpt-neox && \
        python3 tools/preprocess_data.py \
            --input {configs} --output-prefix {path} \
            --tokenizer-type HFTokenizer \
            --vocab-file /mnt/ssd-1/data/20B_tokenizer.json \
            --workers 95\
            --append-eod"
    
    print(f"tokenizing {key} dataset")
    
    subprocess.run(
        exec_command,
        shell=True
    )

def load_from_json(filename):
    '''Loads an array from json and returns it for multiprocessing'''
    with open(filename) as f:
        arr = json.load(f)
    
    print(f"Loaded {filename}")
    return arr
if __name__ == '__main__':
    SEED = 1234 # seed used in shuffling to sample items
    lock = multiprocessing.Manager().Lock()
    with open('processed_configs.txt') as f: # convert documents to json
        configs = f.read().splitlines()
    futures = []
    with ProcessPoolExecutor(95) as p: 
        for config in configs:
            futures.append(p.submit(process,config,lock,))
    
    for future in futures:
        future.result()   
    
    global PROCESSED_DATASET_DICT # combine individual jsons into a specific config
    PROCESSED_DATASET_DICT = {}
    for objective in DATASETS:
        for config in DATASETS[objective]:
            PROCESSED_DATASET_DICT[config] = []
        
    PROCESSED_DATASET_DICT['test'] = []
    PROCESSED_DATASET_DICT['validation'] = []
    PROCESSED_DATASET_DICT['others'] = []

    with ProcessPoolExecutor(95) as executor:
        futures = {}
        for key in PROCESSED_DATASET_DICT:
            futures[key] = []
            for (dirpath,dirs,filepaths) in os.walk(f'./jsondata/{key}/'):
                for filepath in filepaths:
                    futures[key].append(executor.submit(load_from_json,os.path.join(dirpath,filepath)))
            
    for key in futures:
        for future in futures[key]:
            PROCESSED_DATASET_DICT[key].extend(future.result())
    
    for key in PROCESSED_DATASET_DICT: # Resulting data
        print(key,len(PROCESSED_DATASET_DICT[key]))
    
    for key in PROCESSED_DATASET_DICT.keys(): # Sampling and archiving
        path = f'/mnt/ssd-cluster/P3_configs/{key}'
        if(os.path.exists(path)):
            shutil.rmtree(path)
        ar = lm_dataformat.Archive(path)
        if(key in ['test','validation','others']):
            
            for text in PROCESSED_DATASET_DICT[key]:
                ar.add_data(text)
        else:
            random.seed(SEED)
            random.shuffle(PROCESSED_DATASET_DICT[key])
            # for i in range(min(len(PROCESSED_DATASET_DICT[key]),500000)): for sampled dataset
            for i in range(len(PROCESSED_DATASET_DICT[key])):
                text = PROCESSED_DATASET_DICT[key][i]
                ar.add_data(text)

        ar.commit()
    
    
    for key in ['train','test','validation','others']: # Tokenizing
        tokenize(key)
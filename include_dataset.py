from datasets import get_dataset_config_names, load_dataset, load_from_disk
import re
from p3_transform import tokenize


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
        # 'hellaswag',
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


def to_include(config):
    '''Checks if a path can be included in the train/test/validation dataset'''
    for section in DATASETS.values():
        for pattern in section:
            if(re.search(pattern,config)):
                return True
    
    return False


if __name__ == '__main__':
    files = {}
    for key in ['train','test','validation']:
        files[key] = open(f'{key}_paths.txt','r').read().splitlines()
    
    processed_files = {}
    for key,configs in files.items():
        processed_files[key] = []
        for config in configs:
            if(to_include(config)):
                processed_files[key].append(config)
    
    for key,configs in processed_files.items():
        tokenize(configs,key)
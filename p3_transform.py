from datasets import load_from_disk, get_dataset_config_names
import os
import subprocess
import lm_dataformat
import shutil
from multiprocess import Pool,Lock


def process_text(text):
    '''Strips the text from any spaces (or) new lines to maintain consistancy'''
    text = text.strip(' \n')
    return text

def transform(config,path='/home/mchorse/P3/data/',savepath='/home/mchorse/P3/jsonldata'):
    '''Converts a Huggingface P3 dataset to compressed jsonl format'''
    
    
    
    
    lock.acquire()
    hf_ds = load_from_disk(f'/home/mchorse/P3/data/{config}')
    lock.release()

    print(f'Processing {config}')
    created_file_names = {}
    for key in hf_ds.keys():
        ar = lm_dataformat.Archive(f'{savepath}/{config}/{key}')
        
        for batch in hf_ds[key]:
            if('is_correct' in batch and batch['is_correct'] == False):
                continue
            
            prompt = process_text(batch['inputs_pretokenized'])
            response = process_text(batch['targets_pretokenized'])

            text = f'{prompt}\n{response}'

            ar.add_data(text)
        
        ar.commit()
        

        created_file_names[key] = f'{savepath}/{config}/{key}'

    print(f"Processed {config}")
    lock.acquire()
    for key in created_file_names.keys():
        if(key == 'val'):
            with open(f'validation_paths.txt','a') as f:
                f.write(created_file_names[key])
                f.write('\n')
        else:
            with open(f'{key}_paths.txt','a') as f:
                f.write(created_file_names[key])
                f.write('\n')
    with open('processed_configs.txt','a') as f:
        f.write(config + '\n')
    lock.release()

def init(l):
    global lock
    lock = l

def tokenize(paths,key):
    '''Converts compressed jsonl paths to megatron compatible tokenized format'''
    if not os.path.exists('/mnt/ssd-cluster/P3_combined/'):
        os.mkdir('/mnt/ssd-cluster/P3_combined/')
    
    subprocess.run(
            f"cd /home/mchorse/gpt-neox && python3 tools/preprocess_data.py --input {','.join(paths)} --output-prefix /mnt/ssd-cluster/P3_combined/{key} --tokenizer-type HFTokenizer --vocab-file /mnt/ssd-1/data/20B_tokenizer.json --workers 40",
            shell=True
        )
if __name__ == '__main__':

    savepath = '/home/mchorse/P3/jsonldata'
    if(not os.path.exists(savepath)):
        os.mkdir(savepath)
    
    
    l = Lock()
    with open('processed_configs.txt','r') as f:
        processed_configs = f.read().splitlines()
    with Pool(40,initializer = init,initargs=(l,)) as p:
        p.map(transform,[i for i in get_dataset_config_names('bigscience/P3') if i not in processed_configs])
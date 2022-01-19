from datasets import load_from_disk, get_dataset_config_names
import os
import subprocess
import lm_dataformat
import shutil
from multiprocess import Pool


def process_text(text):
    '''Strips the text from any spaces (or) new lines to maintain consistancy'''
    text = text.strip(' \n')
    return text

def transform(config,path='/home/mchorse/P3/data/',savepath='/home/mchorse/P3/jsonldata'):
    '''Converts a Huggingface P3 dataset to tokenized format'''
    
    print(f'Processing {config}')
    
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    hf_ds = load_from_disk(f'./data/{config}')
    os.mkdir(f"/mnt/ssd-cluster/P3/{config}/")
    
    created_file_names = []
    for key in hf_ds.keys():
        ar = lm_dataformat.Archive(f'{savepath}/{config}_{key}')
        
        for batch in hf_ds[key]:
            if('is_correct' in batch and batch['is_correct'] == False):
                continue
            
            prompt = process_text(batch['inputs_pretokenized'])
            response = process_text(batch['targets_pretokenized'])

            text = f'{prompt}\n{response}'

            ar.add_data(text)
        
        ar.commit()
        subprocess.run(
            f"cd /home/mchorse/gpt-neox && python3 tools/preprocess_data.py --input {savepath}/{config}_{key} --output-prefix /mnt/ssd-cluster/P3/{config}/{key} --tokenizer-type HFTokenizer --vocab-file /mnt/ssd-1/data/20B_tokenizer.json",
            shell=True,
            capture_output = True
        )

        shutil.rmtree(f'{savepath}/{config}_{key}')

        created_file_names.append(f'/mnt/ssd-cluster/P3/{config}/{key}_text_document')



    print(f'Processed {config}')

    return created_file_names

if __name__ == '__main__':
    transform('super_glue_multirc_confirm')
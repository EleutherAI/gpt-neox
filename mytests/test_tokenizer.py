import sys
sys.path.append('/home/lfsm/code/gpt-neox/')
from megatron.neox_arguments import NeoXArgs

# ymls = ['/home/lfsm/code/gpt-neox/configs/800M.yml', '/home/lfsm/code/gpt-neox/configs/local_setup.yml']
ymls = ['/home/lfsm/code/gpt-neox/configs/20B.yml']

args = NeoXArgs.from_ymls(ymls)
# args.tokenizer_type = 'HFGPT2Tokenizer'
args.build_tokenizer()

text_1 = 'I am a spider man'
text_2 = ['I am a spider man','I am a bat man']
tokenize = args.tokenizer.tokenize
seq_length = args.seq_length
print(tokenize(text_1,seq_length))
print(tokenize(text_1,seq_length).shape)
print(tokenize(text_2,seq_length))
print(tokenize(text_2,seq_length).shape)

import sys
sys.path.append('/ccs/home/lfsm/code/gpt-neox')
from megatron.neox_arguments import NeoXArgs
from megatron.data.data_utils import build_web_train_valid_test_data_iterators

ymls = [r'/ccs/home/lfsm/code/gpt-neox/configs/20B.yml']

neox_args = NeoXArgs.from_ymls(ymls)
neox_args.configure_distributed_args()
neox_args.build_tokenizer()

neox_args.train_data_paths = r'/gpfs/alpine/csc499/proj-shared/LAION-400m-webdataset/data/{41400..41401}.tar'
neox_args.valid_data_paths = r'/gpfs/alpine/csc499/proj-shared/LAION-400m-webdataset/data/{41400..41401}.tar'


from megatron.data.webdataset import get_wds_data
train_data = get_wds_data(neox_args,is_train=True)

train_dataloader = train_data.dataloader

print('start get data')
i=0
for batch in train_dataloader:
    i += 1
    print(i)
    print(batch[0].shape)
    print(batch[1].shape)
    print(batch)
    if i % 10==0:
        print(f"sample {i} times done")
        break
print(f'total sample number is {i}')

val_data = get_wds_data(neox_args,is_train=False)

val_dataloader = val_data.dataloader

print('start get data')
i=0
for batch in val_dataloader:
    i += 1
    print(i)
    print(batch[0].shape)
    print(batch[1].shape)
    print(batch)
    if i % 10==0:
        print(f"sample {i} times done")
        break
print(f'val total sample number is {i}')

train_data_iterator, valid_data_iterator, test_data_iterator = build_web_train_valid_test_data_iterators(neox_args)

print(next(train_data_iterator))
print(next(valid_data_iterator))

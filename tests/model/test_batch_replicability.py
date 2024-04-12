import torch
import numpy as np

""" 
Idea for the verification: save batches per rank on two datasets, one being a mix of say data sources
A+B, and the other being just A. To debug things, we used
A = pile
B = pile+slimp
and B being 50/50 mix of pile and slimp. We save 3 batches for pile+slimp to ensure that even with noise, the sampling
is as deterministic as we might think, the first batch of pile should be in the first 3 batches of pile+slimp.
To save the files used to debug, add in training.py below:
    # Data stuff.
    timers("train/valid/test data iterators").start()
    (
        train_data_iterator,
        valid_data_iterator,
        test_data_iterator,
    ) = build_train_valid_test_data_iterators(neox_args=neox_args)
the following
    num_batches_to_test_replicability = 3
    save_dummy = torch.stack([next(train_data_iterator)["text"] for _ in range(num_batches_to_test_replicability)])
    torch.save(save_dummy[0], "pile_rank_{}.my_repro_batchs".format(neox_args.rank))
    print(save_dummy[0])
    # print(next(train_data_iterator), "\npile_slimp1")
    # torch.distributed.barrier()
    # print(next(train_data_iterator), "\npile_slimp2")
    # torch.distributed.barrier()
    # print(next(train_data_iterator), "\npile_slimp3")
    torch.distributed.barrier()
    exit(0)
you need 2 runs of this with the two datasets you consider. Don't forget to change the filename of the torch.save()
"""

num_ranks = 6
dataset_A = [None] * num_ranks
dataset_B = [None] * num_ranks
dataset_A_name = "pile_replay"
dataset_B_name = "pile+slimp"
# Use 0 to use all batches saved. If only one batch was saved, this param gets ignored.
num_batches_A = 3
num_batches_B = 3


# cat all batches in a limit of num_batches
def cat_only_process_num_batches(dataset, num_ranks, num_batches=0):
    dim = 1 if len(dataset[0].shape)==3 \
            else 0
    # only use num_batches batches, if tensor has shape [num_batches, sample_idx_in_batch, seq_len]
    if num_batches and dim:
        new_shape_format = [-1, dataset[0].shape[-1]]
        return (torch.cat([dataset[i][:num_batches] for i in range(num_ranks)], dim=dim)).view(new_shape_format)
    else:
        return torch.cat([dataset[i] for i in range(num_ranks)], dim=dim)

for i in range(num_ranks):
    dataset_A[i] = torch.load("gpt-neox/{}_rank_{}.adam".format(dataset_A_name, i))
    dataset_B[i] = torch.load("gpt-neox/{}_rank_{}.adam".format(dataset_B_name, i))

dataset_A_cat = cat_only_process_num_batches(dataset_A, num_ranks, num_batches=num_batches_A)
dataset_B_cat = cat_only_process_num_batches(dataset_B, num_ranks, num_batches=num_batches_B)
# dataset_B_cat = dataset_B_cat.reshape([-1, dataset_B_cat.shape[-1]])
# dataset_A_cat = torch.cat([dataset_A[i][:num_batches_A] for i in range(num_ranks)], dim=0)

checks = [False] * dataset_A_cat.shape[0]
for i in range(len(checks)):
    checks[i] = torch.all(torch.isin(dataset_A_cat[i], dataset_B_cat)).item()

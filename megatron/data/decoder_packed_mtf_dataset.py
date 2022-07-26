import os
import time

import numpy as np
import torch

from megatron import print_rank_0, mpu, logging
from megatron.data.blendable_dataset import BlendableDataset
from megatron.data.dataset_utils import get_datasets_weights_and_num_samples, get_split_by_range_, \
    get_train_valid_test_split_
from megatron.data.mtf_dataset import MTFDataset
from megatron.data.temp_data_utils import get_indexed_dataset

logger = logging.get_logger(__name__)

def build_train_valid_test_datasets(
    data_prefix,
    data_impl,
    splits_string,
    seq_length: int,
    pad_token: int,
    eos_token: int,
    train_valid_test_num_samples,
    seed,
    skip_warmup
):
    """Build train, valid, and test datasets."""

    # Single dataset.
    if len(data_prefix) == 1:
        all_train_datasets, all_valid_datasets, all_test_datasets = _build_train_valid_test_datasets(
            data_prefix=data_prefix[0],
            data_impl=data_impl,
            splits_string=splits_string,
            seq_length=seq_length,
            pad_token=pad_token,
            eos_token=eos_token,
            train_valid_test_num_samples=train_valid_test_num_samples,
            seed=seed,
            skip_warmup=skip_warmup
        )
    # Blending dataset.
    else:

        output = get_datasets_weights_and_num_samples(data_prefix=data_prefix, train_valid_test_num_samples=train_valid_test_num_samples)
        prefixes, weights, datasets_train_valid_test_num_samples = output

        # Build individual datasets.
        train_datasets = []
        valid_datasets = []
        test_datasets = []
        for i in range(len(prefixes)):
            train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
                data_prefix=prefixes[i],
                data_impl=data_impl,
                splits_string=splits_string,
                seq_length=seq_length,
                pad_token=pad_token,
                eos_token=eos_token,
                train_valid_test_num_samples=datasets_train_valid_test_num_samples[i],
                seed=seed,
                skip_warmup=skip_warmup
            )
            if train_ds:
                train_datasets.append(train_ds)
            if valid_ds:
                valid_datasets.append(valid_ds)
            if test_ds:
                test_datasets.append(test_ds)

        all_train_datasets = BlendableDataset(train_datasets, weights) \
                            if train_datasets else None
        all_valid_datasets = BlendableDataset(valid_datasets, weights) \
                            if valid_datasets else None
        all_test_datasets = BlendableDataset(test_datasets, weights) \
                            if test_datasets else None

    return all_train_datasets, all_valid_datasets, all_test_datasets


def build_dataset_group(
    dataset_group_name,
    paths,
    weights,
    splits,
    data_impl,
    seq_length: int,
    pad_token: int,
    eos_token: int,
    train_valid_test_num_samples,
    seed,
    skip_warmup,
    train_valid_test
):
    '''
    Build a single dataset group corresponding to Option 2 of data loading see arguments.py
    a dataset group is passed in the following form
    GIVEN_NAME WEIGHT1 START:END PATH1, WEIGHT2 START:END PATH2, WEIGHT2 START:END PATH2
    or alternatively
    GIVEN_NAME PATH1    # for a single dataset to be used fully
    '''

    assert train_valid_test in ["train","valid","test"]

    # Single dataset.
    if len(paths) == 1:
        dataset = _build_single_datasets(
            data_prefix=paths[0],
            range_string=splits[0],
            data_impl=data_impl,
            seq_length=seq_length,
            pad_token=pad_token,
            eos_token=eos_token,
            train_valid_test_num_samples=train_valid_test_num_samples,
            seed=seed,
            skip_warmup=skip_warmup,
            dataset_group_name=dataset_group_name,
            train_valid_test=train_valid_test
        )
        return dataset
    # Blending dataset.
    else:

        data_prefix = []
        # data_prefix is of the shape:
        # ["WEIGHT1", "PATH1", "WEIGHT2", "PATH2", "WEIGHT3", "PATH3"]
        for w,p in zip(weights, paths):
            data_prefix += [w,p]

        output = get_datasets_weights_and_num_samples(data_prefix,
                                                    train_valid_test_num_samples)
        prefixes, weights, datasets_train_valid_test_num_samples = output

        # Build individual datasets.
        datasets = []
        for i in range(len(prefixes)):
            ds = _build_single_datasets(
                data_prefix=prefixes[i],
                range_string=splits[i],
                data_impl=data_impl,
                seq_length=seq_length,
                pad_token=pad_token,
                eos_token=eos_token,
                train_valid_test_num_samples=datasets_train_valid_test_num_samples[i],
                seed=seed,
                skip_warmup=skip_warmup,
                dataset_group_name=dataset_group_name,
                train_valid_test=train_valid_test
            )

            datasets.append(ds)
        all_datasets = BlendableDataset(datasets, weights)

        return all_datasets

def _build_single_datasets(
    data_prefix,
    range_string,
    data_impl,
    seq_length: int,
    pad_token: int,
    eos_token: int,
    train_valid_test_num_samples,
    seed,
    skip_warmup,
    dataset_group_name,
    train_valid_test
):
    """Build a single dataset"""

    assert train_valid_test in ["train","valid","test"]
    index = ["train","valid","test"].index(train_valid_test)

    # Target indexed dataset.
    target_indexed_dataset = get_indexed_dataset(
        data_prefix=data_prefix,
        is_input=False,
        data_impl=data_impl,
        skip_warmup=skip_warmup
    )

    total_num_of_documents = target_indexed_dataset.sizes.shape[0]
    # this corresponds to option2 for data loading on the form
    # WEIGHT1 START:END PATH1, WEIGHT2 START:END PATH2, WEIGHT3 START:END PATH3
    # splits here is an array of size 2  [start_index, end_index]
    splits = get_split_by_range_(range_string=range_string, size=total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    print_rank_0('    {}:'.format(dataset_group_name))
    print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[0], splits[1],
                                        splits[1] - splits[0]))

    def build_dataset(name):
        dataset = None
        if splits[1] > splits[0]:
            documents = np.arange(start=splits[0], stop=splits[1],
                                  step=1, dtype=np.int32)
            dataset = DecoderPackedMTFDataset(
                name=name,
                data_prefix=data_prefix,
                data_impl=data_impl,
                skip_warmup=skip_warmup,
                documents=documents,
                seq_length=seq_length,
                pad_token=pad_token,
                eos_token=eos_token,
                num_samples=train_valid_test_num_samples[index],
                seed=seed
            )
        return dataset

    dataset = build_dataset(dataset_group_name)

    return dataset


def _build_train_valid_test_datasets(
    data_prefix,
    data_impl,
    splits_string,
    seq_length: int,
    pad_token: int,
    eos_token: int,
    train_valid_test_num_samples,
    seed,
    skip_warmup
):
    """Build train, valid, and test datasets."""

    # Target indexed dataset.
    target_indexed_dataset = get_indexed_dataset(data_prefix, is_input=False, data_impl=data_impl, skip_warmup=skip_warmup)

    total_num_of_documents = target_indexed_dataset.sizes.shape[0]
    # splits here is an array of size 4  [train_start_index, valid_start_index, test_start_index, test_end_index]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)
    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    def print_split_stats(name, index):
        print_rank_0('    {}:'.format(name))
        print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[index], splits[index + 1],
                                        splits[index + 1] - splits[index]))
    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index], stop=splits[index + 1],
                                  step=1, dtype=np.int32)
            dataset = DecoderPackedMTFDataset(
                name=name,
                data_prefix=data_prefix,
                data_impl=data_impl,
                skip_warmup=skip_warmup,
                documents=documents,
                seq_length=seq_length,
                pad_token=pad_token,
                eos_token=eos_token,
                num_samples=train_valid_test_num_samples[index],
                seed=seed
            )
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)


class DecoderPackedMTFDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        name,
        data_prefix,
        data_impl,
        skip_warmup,
        documents,
        num_samples,
        seq_length: int,
        pad_token: int,
        eos_token: int,
        seed,
    ):
        self.mtf_dataset = MTFDataset(name=name, data_prefix=data_prefix, data_impl=data_impl, skip_warmup=skip_warmup, documents=documents)

        self.pad_token = pad_token
        self.seq_length = seq_length

        self.sample_index, self.shuffle_index = _build_index_mappings(name=name, data_prefix=data_prefix, nb_documents=len(documents), mtf_dataset=self.mtf_dataset, num_samples=num_samples, seq_length=seq_length, seed=seed)

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        # Get the shuffled index.
        start, end = self.sample_index[idx]
        mtf_samples_indices = self.shuffle_index[start: end]
        # TODO @thomasw21 build a dataset that generates an entire batch instead of a row (allows for more optimization)
        items = [self.mtf_dataset[sample_id] for sample_id in mtf_samples_indices]

        return self.pack_samples(items)

    def pack_samples(self, items):
        """
        Greedily packs samples.
        Items:
            [
                {
                    'input_tokens': array([6, 7]),
                    'target_tokens': array([8])
                },
                {
                    'input_tokens': array([3, 4]),
                    'target_tokens': array([5])
                }
            ]
        Output:
            decoder_tokens = [[6, 7, 8, 3, 4, 5, <pad>]]: Concatenation of tokens followed with padding tokens.
            decoder_segment_ids = [[1, 1, 1, 2, 2, 2, 0]]: Segment ids determine original documents.
            decoder_is_inputs = [[1, 1, 0, 1, 1, 0, 0]]: `1` depicts inputs, `0` depicts target.
        """

        decoder_tokens = np.full((self.seq_length,), self.pad_token, dtype=np.int64)
        decoder_segment_ids = np.zeros((self.seq_length,), dtype=np.int64)
        decoder_is_inputs = np.full((self.seq_length,), False, dtype=bool)

        # `0` is reserved for padding
        item_num = 1
        cur_len = 0

        assert len(items) > 0

        for token_dict in items:
            input_token_len = len(token_dict["input_tokens"])
            target_token_len = len(token_dict["target_tokens"])

            total_len = input_token_len + target_token_len

            if cur_len + total_len > self.seq_length:
                # This should not happen at the indexing should only allow the correct number of items
                raise ValueError(f"""Items to be packed do not fit inside a single sample.
                    current length: {cur_len}
                    input tokens length: {input_token_len}
                    target token length: {target_token_len}
                    expected sequence length: {self.seq_length}
                """)

            decoder_tokens[cur_len: cur_len + input_token_len] = token_dict["input_tokens"]
            decoder_tokens[cur_len + input_token_len: cur_len + total_len] = token_dict["target_tokens"]
            decoder_segment_ids[cur_len: cur_len + total_len] = item_num
            decoder_is_inputs[cur_len: cur_len + input_token_len] = 1  # inputs
            # targets are already 0 at init, no need to update `decoder_is_inputs`

            item_num += 1
            cur_len += total_len
            assert cur_len <= self.seq_length

        return {
            "decoder_token_ids": decoder_tokens,
            "decoder_segment_ids": decoder_segment_ids,
            "decoder_is_inputs": decoder_is_inputs,
        }


def _build_index_mappings(
    name,
    data_prefix,
    nb_documents,
    mtf_dataset,
    num_samples: int,
    seq_length: int,
    seed,
):
    """
    - `shuffle_index` is [num_epoch * len(self.mtf)]
    - `sample_index` is [num_sample, 2] (storing the start and end of the sample). We query the sample via `self.shuffle_index[start:end]`
    TODO @thomas21 Instead of loading individually samples, we save the packing one and for all
    """
    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    _filename = data_prefix
    _filename += '_{}_indexmap'.format(name)
    _filename += '_{}ns'.format(num_samples)
    _filename += '_{}s'.format(seed)
    sample_idx_filename = _filename + '_decoder_packed_batch_idx.npy'
    shuffle_idx_filename = _filename + '_decoder_packed_shuffle_idx.npy'

    # Build the indexed mapping if not exist.
    if torch.distributed.get_rank() == 0:
        if (not os.path.isfile(sample_idx_filename)) or \
           (not os.path.isfile(shuffle_idx_filename)):

            print_rank_0(' > WARNING: could not find index map files, building '
                         'the indices on rank 0 ...')

            # iteratively add the entire dataset for every epoch and see if it's enough given current packing strategy
            start_time = time.time()
            row_offset = 0
            old_sample_start = 0
            epoch = 0
            shuffle_idx = []
            sample_idx = []
            while len(sample_idx) <= num_samples:
                new_document_ids = _build_shuffle_idx(nb_documents=nb_documents, np_rng=np_rng)
                # Generate a shuffling of the entire dataset
                shuffle_idx.append(new_document_ids)
                # Packs them into a single sample
                new_samples, row_offset, old_sample_start = _build_sample_idx(
                    mtf_dataset=mtf_dataset,
                    document_ids=new_document_ids,
                    seq_length=seq_length,
                    row_offset=row_offset,
                    old_sample_start=old_sample_start,
                    epoch=epoch
                )
                sample_idx.extend(new_samples)
                epoch += 1

            shuffle_idx = np.concatenate(shuffle_idx, axis=0)
            sample_idx = np.stack(sample_idx, axis=0)

            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
            np.save(sample_idx_filename, sample_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save shuffle-idx and sample-idx mapping'
                         ' (seconds): {:4f}'.format(time.time() - start_time))

    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    counts = torch.cuda.LongTensor([1])
    torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
    torch.distributed.all_reduce(counts, group=mpu.get_pipeline_model_parallel_group())
    assert counts[0].item() == (
        torch.distributed.get_world_size() //
        torch.distributed.get_world_size(group=mpu.get_tensor_model_parallel_group()))

    # Load mappings.
    start_time = time.time()
    print_rank_0(' > loading doc-idx mapping from {}'.format(
        sample_idx_filename))
    sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0(' > loading shuffle-idx mapping from {}'.format(
        shuffle_idx_filename))
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(
        time.time() - start_time))

    return sample_idx, shuffle_idx

def _build_sample_idx(mtf_dataset, document_ids, seq_length, row_offset, old_sample_start, epoch):
    """Build start and off index of each `full` batch, return that list of batch + start of the unfinished batch"""
    row_length = row_offset

    full_samples = []
    current_sample_start = old_sample_start
    epoch_offset = epoch * len(document_ids)

    assert epoch_offset >= current_sample_start
    for current_sample_end, document_id in enumerate(document_ids):
        current_sample_end = epoch_offset + current_sample_end
        sample_sizes = mtf_dataset.size(document_id)

        # TODO @thomasw21 figure out if we add <eos> tokens
        tok_len = sample_sizes["input_tokens"] + sample_sizes["target_tokens"]

        row_length = row_length + tok_len
        if row_length > seq_length:
            # current sample can't be added and requires to be added in the next one
            if current_sample_end > current_sample_start:
                full_samples.append(np.asarray([current_sample_start, current_sample_end]))
            current_sample_start = current_sample_end
            row_length = tok_len

            if tok_len > seq_length:
                # TODO @thomasw21 handle the case where a single sample cannot fit inside a row. We can
                #   - silently skip that value [currently implemented]
                #   - truncate to `seq_length`, and keep the right part
                logger.warning(f"Skipping sample id={document_id}. Maximum sequence length: {seq_length}, sample length: {tok_len}")
                current_sample_start = current_sample_end + 1  # skipping
                row_length = 0
                continue

    return full_samples, row_length, current_sample_start

def _build_shuffle_idx(nb_documents: int, np_rng):
    """Build the range [0, dataset_size) and shuffle."""
    dtype_ = np.int64

    result = np.arange(start=0, stop=nb_documents, step=1, dtype=dtype_)

    # in-place shuffling
    np_rng.shuffle(result)

    return result

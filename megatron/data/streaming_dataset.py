try:
    from streaming import Stream, StreamingDataset
    from streaming.base.world import World
except ModuleNotFoundError:
    raise Exception("Must install `streaming` package to use StreamingDatasets!")

from typing import Optional, Sequence, Union, Any, Dict, List
from megatron import print_rank_0

import torch
import numpy as np

import base64

# TAKEN FROM MOSAICML LLM-FOUNDRY
# https://github.com/mosaicml/llm-foundry/blob/main/llmfoundry/data/text_data.py#L23C1-L192C28
class StreamingTextDataset(StreamingDataset):
    """Generic text dataset using MosaicML's StreamingDataset.

    Args:
        max_seq_len (int): The max sequence length of each sample.
        streams (Sequence[Stream], optional): One or more Streams to stream/cache samples from,
            which may be upsampled or downsampled. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        remote (str, optional): Remote path or directory to download the dataset from. If ``None``,
            its data must exist locally. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        local (str, optional): Local working directory to download shards to. This is where shards
            are cached while they are being used. Uses a temp directory if not set.
            StreamingDataset uses either ``streams`` or ``remote``/``local``. Defaults to ``None``.
        split (str, optional): Which dataset split to use, if any. If provided, we stream from/to
            the ``split`` subdirs of  ``remote`` and ``local``. Defaults to ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        keep_zip (bool): Whether to keep or delete the compressed form when decompressing
            downloaded shards. If ``False``, keep iff remote is local or no remote. Defaults to
            `False``.
        epoch_size (Union[int, str], optional): Number of samples to draw per epoch balanced across all
            streams. If ``None``, takes its value from the total number of underlying samples.
            Provide this field if you are weighting streams relatively to target a larger or
            smaller epoch size. Defaults to ``None``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. If ``None``, its value is set to ``8 * batch_size``. Defaults to ``None``.
        cache_limit (Union[int, str], optional) - Maximum size in bytes of this StreamingDataset's
            shard cache. Before downloading a shard, the least recently used resident shard(s) may
            be evicted (deleted from the local cache) in order to stay under the limit. Set to None
            to disable shard eviction. Supports integer bytes as well as string human-readable
            bytes (e.g., 100b, 64kb, 77mb, and so on). Defaults to None.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``orig``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with
            resumption. If ``None``, this is interpreted as 64 times the number of physical
            nodes of the initial run if ``shuffle_algo`` is ``py1s`` or ``py2s``, and simply the
            number of physical nodes of the initial run otherwise. Defaults to ``None``.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1e``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
        shuffle_block_size (int, optional): Unit of shuffle. A canonical node's samples are split
            into blocks of this size, and samples within each block are shuffled. If ``None``, its
            value is calculated as ``max(4_000_000 // num_canonical_nodes), 1 << 18)``. Defaults to
            ``None``.
        sampling_method (str): Which sampling method to use, either ``balanced`` or ``fixed``.
            Defaults to ``balanced``.
        sampling_granularity (int): When picking samples for a stream's final partial repeat,
            how many samples to pick from the same shard at a time (``1`` for evenly balanced
            across shards, ``1000`` to pick 1000 samples from the same shard at a time, etc).
            Defaults to ``1``.
        batching_method (str): Which batching method to use, either ``random``, ``stratified``, or
            ``per_stream``. Defaults to ``random``.
    """

    def __init__(self,
                 max_seq_len: int,
                 streams: Optional[Sequence[Stream]] = None,
                 remote: Optional[str] = None,
                 local: Optional[str] = None,
                 split: Optional[str] = None,
                 download_retry: int = 2,
                 download_timeout: float = 60,
                 validate_hash: Optional[str] = None,
                 keep_zip: bool = False,
                 epoch_size: Optional[Union[int, str]] = None,
                 predownload: Optional[int] = None,
                 cache_limit: Optional[Union[int, str]] = None,
                 partition_algo: str = 'relaxed',
                 num_canonical_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 shuffle: bool = False,
                 shuffle_algo: str = 'py1e',
                 shuffle_seed: int = 9176,
                 shuffle_block_size: Optional[int] = None,
                 sampling_method: str = 'balanced',
                 sampling_granularity: int = 1,
                 batching_method: str = 'random',
                 **kwargs: Any):

        group_method = kwargs.pop('group_method', None)
        if group_method is not None:
            raise NotImplementedError(
                'group_method is deprecated and has been removed.\nTo ' +
                'concatenate, use the --concat_tokens ' +
                'argument when creating your MDS dataset with concat_c4.py')

        if len(kwargs) > 0:
            raise ValueError(
                f'StreamingTextDataset() got an unexpected keyword argument: {kwargs}'
            )

        if local is not None and (remote is None or (local == remote)):
            if os.path.isdir(local):
                contents = set(os.listdir(local))
                if split not in contents:
                    raise ValueError(
                        f'local directory {local} does not contain split {split}'
                    )

        # TODO: discover where yamls are being converted incorrect, but temporary workaround
        if isinstance(shuffle_block_size, float):
            shuffle_block_size = int(shuffle_block_size)

        # Build Dataset
        super().__init__(
            streams=streams,
            remote=remote,
            local=local,
            split=split,
            download_retry=download_retry,
            download_timeout=download_timeout,
            validate_hash=validate_hash,
            keep_zip=keep_zip,
            epoch_size=epoch_size,
            predownload=predownload,
            cache_limit=cache_limit,
            partition_algo=partition_algo,
            num_canonical_nodes=num_canonical_nodes,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_algo=shuffle_algo,
            shuffle_seed=shuffle_seed,
            shuffle_block_size=shuffle_block_size,
            sampling_method=sampling_method,
            sampling_granularity=sampling_granularity,
            batching_method=batching_method,
        )

        self.max_seq_len = max_seq_len

    def _read_binary_tokenized_sample(self, sample: Dict[str,
                                                         Any]) -> torch.Tensor:
        return torch.from_numpy(
            np.frombuffer(sample['tokens'],
                          dtype=np.int64)[:self.max_seq_len].copy())

    # How to process a sample
    def __getitem__(self,
                    idx: int) -> Union[Dict[str, List[int]], torch.Tensor]:
        sample = super().__getitem__(idx)
        if 'tokens' in sample:
            token_sample = self._read_binary_tokenized_sample(sample)
        else:
            raise RuntimeError(
                'StreamingTextDataset needs samples to have a `tokens` column'
            )
        #print(token_sample.shape)
        return token_sample


def build_streaming_dataset(split, neox_args=None):
    """build a StreamingTextDataset"""

    assert split in ["train", "valid", "test"]

    train_iters = neox_args.train_iters
    eval_iters = (train_iters // neox_args.eval_interval + 1) * neox_args.eval_iters
    test_iters = neox_args.eval_iters
    train_val_test_num_samples = {
        "train": train_iters * neox_args.train_batch_size,
        "valid": eval_iters * neox_args.train_batch_size,
        "test": test_iters * neox_args.train_batch_size,
    }

    data_paths = {
        "train": neox_args.train_data_paths, 
        "valid": neox_args.valid_data_paths, 
        "test": neox_args.test_data_paths
    }[split]


    data_weights = {
        "train": neox_args.train_data_weights,
        "valid": neox_args.valid_data_weights,
        "test": neox_args.test_data_weights,
    }[split]

    if data_weights:
        # normalize proportions
        data_weights = [weight / sum(data_weights) for weight in data_weights]

    streams = []
    import os
    for i, path in enumerate(data_paths): 
        remote = path if "s3://" in path else None
        local=path if "s3://" not in path else f"/tmp/{path[5:]}"
        print_rank_0(f"stream {i} remote: {remote} local: {local}")
        streams.append(
            Stream(
                remote=path if "s3://" in path else None,
                local=path if "s3://" not in path else f"/weka/hailey/cond-training/streaming-cache/{path[5:]}-rank{os.environ['RANK']}",
                proportion=data_weights[i] if data_weights else None, # support for upsampling
            )
        )
    from megatron import mpu   

    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    global_batch_size = neox_args.batch_size * world_size

    # We compute num_canonical_nodes ourselves to
    # be able to save it easily in the config
    # (it's required to resume shuffling correctly)
    shuffle_algo = "py1e" #StreamingDataset default
    num_nodes = World.detect().num_nodes
    print_rank_0("NUM NODES", num_nodes)
    # similar to StreamingDataset code
    num_canonical_nodes = 64 * num_nodes if shuffle_algo in ["py1s", "py2s"] else num_nodes
 
    return StreamingTextDataset(
        max_seq_len=neox_args.seq_length + 1,
        streams=streams,
        split=None,
        epoch_size=train_val_test_num_samples[split],
        # download_timeout=300,
        predownload=8192,
        batch_size=global_batch_size,
        shuffle=True,
        shuffle_seed=neox_args.seed,
        shuffle_algo=shuffle_algo,
        num_canonical_nodes=num_canonical_nodes,
        #batch_size=neox_args.train_micro_batch_size_per_gpu,
    )


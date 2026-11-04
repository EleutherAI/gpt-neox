"""Separate clean/flagged token streams, mixed without padding or token copies."""
import json
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset


class HomogeneousPackedDataset(Dataset):
    def __init__(self, indexed_dataset, directory, seq_length, num_samples, building=False):
        self.indexed_dataset = indexed_dataset
        root = Path(directory)
        meta = json.loads((root/('BUILDING.json' if building else 'COMPLETE.json')).read_text())
        if meta['seq_length'] != seq_length or meta['num_samples'] < num_samples:
            raise ValueError('Homogeneous sampler does not cover this training horizon')
        if meta['documents'] != len(indexed_dataset.sizes):
            raise ValueError('Homogeneous sampler corpus size differs')
        self.seq_length = seq_length
        self.clean_samples = meta['samples_per_route'][0]
        self.order = np.load(root/'sample_order.npy', mmap_mode='r', allow_pickle=False)
        self.streams = [(np.load(root/f'doc_ids_{i}.npy', mmap_mode='r', allow_pickle=False),
                         np.load(root/f'offsets_{i}.npy', mmap_mode='r', allow_pickle=False))
                        for i in (0, 1)]
        if len(self.order) != meta['num_samples']:
            raise ValueError('Incomplete homogeneous sample order')
        for route, (ids, offsets) in enumerate(self.streams):
            if len(offsets) != len(ids)+1 or offsets[0] != 0:
                raise ValueError('Invalid homogeneous document offsets')
            if meta['samples_per_route'][route]*seq_length+1 > offsets[-1]:
                raise ValueError('Homogeneous stream too short')

    def __len__(self):
        return len(self.order)

    def __getitem__(self, index):
        sample = int(self.order[index])
        route = int(sample >= self.clean_samples)
        local = sample - (self.clean_samples if route else 0)
        start = local*self.seq_length
        end = start+self.seq_length+1
        ids, offsets = self.streams[route]
        first = int(np.searchsorted(offsets, start, side='right')-1)
        pieces = []
        pos = start
        while pos < end:
            length = min(end, int(offsets[first+1]))-pos
            if length:
                pieces.append(self.indexed_dataset.get(int(ids[first]),
                              offset=pos-int(offsets[first]), length=length))
                pos += length
            first += 1
        text = np.concatenate(pieces).astype(np.int64, copy=False)
        if len(text) != self.seq_length+1:
            raise ValueError('Truncated homogeneous sample')
        return dict(text=text, gram_flag=np.int64(route))

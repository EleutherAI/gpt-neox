import torch
from torch.utils.data import Dataset, IterableDataset
from .data_utils import get_tokenizer, natural_sort, consume
import random
import glob
import tensorflow as tf
import re
import logging
from itertools import cycle, chain
import os


class GPT2Dataset(IterableDataset):

    def __init__(self, glob_pattern, seq_len, seed=1, shuffle_input_filenames=True, pretokenized=True,
                 filetype="tfrecords", mode="chunks", train=True, tokenizer=None, save_progress_every=None,
                 checkpoint_path=None, resume_from_checkpoint=True):
        self.files = glob.glob(glob_pattern)  # glob pattern pointing to files
        # parses the length of the files, either by encoding in the filenames or by iterating over them
        self._get_lens()
        self.seed = seed  # random seed for shuffling
        if shuffle_input_filenames:
            random.seed(self.seed)
            random.shuffle(self.files)
        else:
            self.files = natural_sort(self.files)
        self.seq_len = seq_len
        self.mode = mode
        self.pretokenized = pretokenized
        if not self.pretokenized:
            raise NotImplementedError  # TODO: tokenize text data on the fly
        implemented_modes = ["chunks"]
        if self.mode not in implemented_modes:
            raise NotImplementedError
        self.filetype = filetype
        implemented_filetypes = ["tfrecords"]
        if self.filetype not in implemented_filetypes:
            raise NotImplementedError
        self.train = train
        self.idx = 0
        self.save_progress_every = save_progress_every
        self.checkpoint_path = checkpoint_path
        self.resume_from_checkpoint = resume_from_checkpoint
        if self.save_progress_every is not None:
            assert self.checkpoint_path is not None

    def _get_number_of_documents(self, filename):
        # extracts number of files from a filename formatted "<name>_<num_documents>.{filetype}."
        # if no pattern is matched, returns None
        match = re.search("_(\d{1,})." + self.filetype + "$", filename)
        return int(match.group(1)) if match is not None else match

    def _get_number_of_documents_by_iteration(self, filename):
        # extracts number of files from a tfrecord document in the event it doesn't have metadata in the filename
        # this could be very slow.
        logging.warning(
            "Found no metadata found in filename - iterating through first tfrecord to find global length")
        count = 0
        if self.filetype == "tfrecords":
            for _ in tf.io.tf_record_iterator(filename):
                count += 1
        return count

    def _get_lens(self):
        lens = []
        for f in self.files:
            n_documents = self._get_number_of_documents(f)
            if n_documents is None:
                n_documents = self._get_number_of_documents_by_iteration(f)
            lens.append(n_documents)
        self.lens = lens

    def _parse_single_example(self, example):
        data = tf.train.Example.FromString(example)
        data = torch.tensor(list(data.features.feature["text"].int64_list.value), dtype=torch.long)
        if self.mode == "chunks":
            assert data.size(0) == self.seq_len + 1
        return data

    def _save_progress(self):
        with open(self.checkpoint_path, "w") as f:
            f.write(str(self.idx))

    def _resume_from_checkpoint(self):
        if os.path.isfile(self.checkpoint_path):
            with open(self.checkpoint_path, "r") as f:
                idx = int(f.read())
        else:
            idx = 0
        if idx >= len(self):  # for > 1 epoch
            idx = len(self) - idx
        return idx

    def _process_tfrecord(self, tfrecords_file):
        for example in tf.io.tf_record_iterator(tfrecords_file):
            self.idx += 1
            if self.save_progress_every is not None:
                if self.idx % self.save_progress_every == 0:
                    self._save_progress()
            yield self._parse_single_example(example)

    def __iter__(self):
        if self.filetype == "tfrecords":
            it = chain.from_iterable(map(self._process_tfrecord, cycle(self.files)))
        else:
            raise NotImplementedError
        if self.resume_from_checkpoint:
            it = consume(it, self._resume_from_checkpoint())
        return it

    def __len__(self):
        sum(self.lens)


class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq

    def __len__(self):
        return self.data.size(0) // self.seq_len

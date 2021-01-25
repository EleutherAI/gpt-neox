import torch
from torch.utils.data import Dataset
from .data_utils import get_tokenizer, natural_sort, skip, FixedSizeOrderedDict
import random
import glob
import tensorflow.compat.v1 as tf
import re
import logging
from itertools import cycle

class GPT2Dataset(Dataset):

    def __init__(self, glob_pattern, seq_len, seed=1, shuffle_input_filenames=True, pretokenized=True,
                 filetype="tfrecords", mode="normal", train=True, tokenizer=None, **kwargs):

        super().__init__()
        self.files = glob.glob(glob_pattern)  # glob pattern pointing to files
        self.seed = seed  # random seed for shuffling

        # shuffle or sort files
        if shuffle_input_filenames:
            random.seed(self.seed)
            random.shuffle(self.files)
        else:
            self.files = natural_sort(self.files)
        self.filetype = filetype  # filetype ["tfrecords"]
        implemented_filetypes = ["tfrecords"]
        if self.filetype not in implemented_filetypes:
            raise NotImplementedError

        self.processed_files = FixedSizeOrderedDict(max=1)  # storage for lazily loading data

        # parses the length of the files, either by encoding in the filenames or by iterating over them
        self._get_lens()

        self.seq_len = seq_len  # set sequence length
        
        self.pretokenized = pretokenized
        if not self.pretokenized:
            raise NotImplementedError  # TODO: tokenize text data on the fly

        self.train = train
        self.mode = mode

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
        self._len = sum(self.lens)

    def _parse_single_example(self, example):
        data = tf.train.Example.FromString(example)
        data = torch.tensor(list(data.features.feature["text"].int64_list.value), dtype=torch.long)
        return data

    def _process_tfrecord(self, tfrecords_file, resume_idx=None):
        for idx, example in enumerate(tf.io.tf_record_iterator(tfrecords_file)):
            yield self._parse_single_example(example)

    def _maybe_process_tfrecord(self, file_idx):
        if self.processed_files.get(file_idx) is None:
            self.processed_files[file_idx] = list(self._process_tfrecord(self.files[file_idx]))
        return self.processed_files[file_idx]

    def _seek(self, idx):
        cumsum = 0
        for count, (f, length) in cycle(enumerate(zip(self.files, self.lens))):
            prev_cumsum = cumsum
            cumsum += length
            if cumsum == idx:
                remainder = 0
                skip_idx = count + 1
                return skip_idx, remainder
            elif cumsum > idx:
                remainder = idx - prev_cumsum
                skip_idx = count
                return skip_idx, remainder

    def __getitem__(self, idx):
        # seek to correct chunk
        seek_idx, remainder = self._seek(idx)
        f = self.files[seek_idx]
        if self.filetype == "tfrecords":
            chunk = self._maybe_process_tfrecord(
                seek_idx)  # parses tfrecord file to a list *once* then stores in memory
        else:
            raise NotImplementedError
        output = chunk[remainder]
        assert output is not None
        assert output.size(0) == (self.seq_len + 1), f"Output shape ({output.size(0)}) != the specified sequence length + 1 ({self.seq_len + 1})"
        if self.mode == "normal":
            return output
        elif self.mode == 'with_labels':
            x_seq = output[:-1]
            y_seq = output[1:]
            return x_seq, y_seq
        else:
            raise ValueError(f'mode {self.mode} not recognized')

    def __len__(self):
        return self._len


class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len, mode="normal"):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        assert mode in ["normal", "with_labels"]
        self.mode = mode

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        if self.mode == "normal":
            full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
            return full_seq
        elif self.mode == "with_labels":
            x_seq = self.data[rand_start: rand_start + self.seq_len].long()
            y_seq = self.data[rand_start+1: rand_start + self.seq_len + 1].long()
            return x_seq, y_seq
        else:
            raise ValueError(f'mode {self.mode} not recognized')

    def __len__(self):
        return self.data.size(0) // self.seq_len

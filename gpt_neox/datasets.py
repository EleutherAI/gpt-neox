import torch
from torch.utils.data import Dataset
from .data_utils import get_tokenizer, natural_sort, skip, FixedSizeOrderedDict
import random
import glob
import tensorflow as tf
import re
import logging
from itertools import cycle
import os
import subprocess
import simdjson as json

class GPT2Dataset(Dataset):

    def __init__(self, glob_pattern, seq_len, seed=1, shuffle_input_filenames=True, pretokenized=True,
                 filetype="tfrecords", mode="chunks", train=True, tokenizer=None, **kwargs):

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

        self.processed_files = FixedSizeOrderedDict(max=2)  # storage for lazily loading data

        # parses the length of the files, either by encoding in the filenames or by iterating over them
        self._get_lens()

        self.seq_len = seq_len  # set sequence length
        self.mode = mode  # set mode ["chunks"]
        implemented_modes = ["chunks"]
        if self.mode not in implemented_modes:
            raise NotImplementedError

        self.pretokenized = pretokenized
        if not self.pretokenized:
            raise NotImplementedError  # TODO: tokenize text data on the fly

        self.train = train

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
        if self.mode == "chunks":
            assert data.size(0) == self.seq_len + 1
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
        return chunk[remainder]  # get item from current chunk

    def __len__(self):
        return self._len


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


class DynamicDataset(Dataset):
    def __init__(self, input_files, tokenizer, max_seq_len, target_field='text', seed=1, shuffle_files=True, **kwargs):
        super().__init__()
        self.files = []
        self.setup_files(input_files)
        if shuffle_files:
            random.seed(seed)
            random.shuffle(self.files)
        self.create_pipeline()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.target_field = target_field
        self.parser = json.Parser()
        self.idx = 0

    def setup_files(self, input_files):
        if isinstance(input_files, str):
            if input_files.endswith('*'):
                self.files = glob.glob(input_files)
            elif os.path.isdir(input_files):
                self.files = glob.glob(os.path.join(input_files, '*'))
        elif isinstance(input_files, list):
            for file_path in input_files:
                if os.path.isfile(file_path) and os.path.exists(file_path):
                    self.files.append(file_path)
                elif file_path.endswith('*'):
                    self.files.extend(glob.glob(file_path))
                elif os.path.isdir(file_path):
                    self.files.extend(glob.glob(os.path.join(file_path, '*')))
        
        self.total_files = len(self.files)
        self.file_idx, self.total_lines = {}, 0
        for file_path in self.files:
            total_lines = self.total_lines_in_file(file_path)
            self.file_idx[file_path] = total_lines
            self.total_lines += total_lines
        logging.info(f'Total Files: {self.total_files}. Total Lines: {self.total_lines}')
    
    def create_pipeline(self):
        self.pipeline = tf.data.TextLineDataset(self.files, num_parallel_reads=tf.data.experimental.AUTOTUNE).as_numpy_iterator()

    def parse_json(self, line):
        try:
            return self.parser.parse(line).as_dict()
        except ValueError:
            return line

    @classmethod
    def total_lines_in_file(cls, file_path):
        return int(subprocess.check_output(['wc', '-l', file_path]).split()[0])
    
    def tokenize_example(self, ex):
        self.idx += 1
        return self.tokenizer(ex[self.target_field], max_length=self.max_seq_len, truncation=True, return_tensors='pt')['input_ids']

    def __getitem__(self, idx):
        try:
            ex = next(self.pipeline)
        except StopIteration:
            del self.pipeline
            self.create_pipeline()
            ex = next(self.pipeline)
        return self.tokenize_example(self.parse_json(ex))

    def __len__(self):
        return self.total_lines

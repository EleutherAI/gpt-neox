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

        self.processed_files = FixedSizeOrderedDict(max=1)  # storage for lazily loading data

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
    
    def _parse_function(self, example_proto):
        features = {
            "text": tf.io.VarLenFeature(tf.int64)
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        return tf.sparse.to_dense(parsed_features["text"], parsed_features["text"].dense_shape[0])

    def _process_tfrecord(self, tfrecords_file, resume_idx=None):
        dataset = tf.data.TFRecordDataset([tfrecords_file])
        dataset = dataset.map(self._parse_function, num_parallel_calls=1)
        for example in dataset.as_numpy_iterator():
            yield torch.tensor(example, dtype=torch.long)

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

class LineSeekableFile:
    def __init__(self, seekable):
        self.fin = seekable
        self.line_map = list()
        self.line_map.append(0)
        while seekable.readline():
            self.line_map.append(seekable.tell())

    def __getitem__(self, index):
        self.fin.seek(self.line_map[index])
        return self.fin.readline()

class DynamicDataset(Dataset):
    def __init__(self, input_files, tokenizer, max_seq_len, target_field='text', seed=1, shuffle_files=True, debug=False, **kwargs):
        super().__init__()
        self.files = []
        self.setup_files(input_files)
        if shuffle_files:
            random.seed(seed)
            random.shuffle(self.files)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.target_field = target_field
        self.token_cache = []
        self.sep_token = tokenizer.eos_token_id
        self.pad_token = tokenizer.pad_token_id
        self.parser = json.Parser()
        self.debug = debug

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
        for x, file_path in enumerate(self.files):
            total_lines = self.total_lines_in_file(file_path)
            self.file_idx[x] = {
                'start': self.total_lines, 'stop': (self.total_lines + total_lines), 
                'file': file_path, 'reader': LineSeekableFile(tf.io.gfile.GFile(file_path, 'rb'))
                }
            if self.debug:
                logging.debug(f'File IDX Start: {self.total_lines} - File IDX End: {self.total_lines + total_lines}')
            self.total_lines += total_lines
        if self.debug:
            logging.debug(f'Total Files: {self.total_files}. Total Lines: {self.total_lines}')
    
    def get_file_line(self, idx):
        for x in range(len(self.files)):
            if idx in range(self.file_idx[x]['start'], self.file_idx[x]['stop']):
                fidx = idx - self.file_idx[x]['start']
                if self.debug:
                    logging.debug(f'File IDX: {fidx}')
                return self.file_idx[x]['reader'][fidx]

    def parse_json(self, line):
        try:
            return self.parser.parse(line).as_dict()[self.target_field]
        except ValueError:
            return line
        except TypeError:
            return line

    @classmethod
    def total_lines_in_file(cls, file_path):
        return int(subprocess.check_output(['wc', '-l', file_path]).split()[0])
    
    def tokenize_example(self, ex):
        if self.token_cache:
            if len(self.token_cache) > self.max_seq_len:
                out = self.token_cache[0:self.max_seq_len]
                tokenized = self.tokenizer(ex)
                self.token_cache = self.token_cache[0:self.max_seq_len].extend(tokenized['input_ids'].append(self.sep_token))
                
            else:
                out = self.token_cache[:]
                self.token_cache = []
                tokenized = self.tokenizer(ex, max_length=(self.max_seq_len - len(out)), truncation=True, return_overflowing_tokens=True)
                out.extend(tokenized['input_ids'])
                if len(out) < self.max_seq_len:
                    _to_pad = self.max_seq_len - len(out)
                    out.extend([self.pad_token for i in range(_to_pad)])
                if tokenized.get('overflowing_tokens', None):
                    self.token_cache = tokenized['overflowing_tokens'].append(self.sep_token)    

        else:
            tokenized = self.tokenizer(ex, max_length=self.max_seq_len, truncation=True, return_overflowing_tokens=True)
            out = tokenized['input_ids']
            if len(out) < self.max_seq_len:
                _to_pad = self.max_seq_len - len(out)
                out.extend([self.pad_token for i in range(_to_pad)])
            if tokenized.get('overflowing_tokens', None):
                self.token_cache = tokenized['overflowing_tokens'].append(self.sep_token)
        
        return torch.tensor(out, dtype=torch.long)

    def __getitem__(self, idx):
        if self.debug:
            logging.debug(f'Getting IDX: {idx}')
        ex = self.get_file_line(idx)
        return self.tokenize_example(self.parse_json(ex.strip()))

    def __len__(self):
        return self.total_lines
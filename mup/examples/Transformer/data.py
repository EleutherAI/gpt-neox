import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = None
        self.valid = None
        self.test = None
        if not self.load_cache(path):
            self.train = self.tokenize(os.path.join(path, 'train.txt'))
            self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
            self.test = self.tokenize(os.path.join(path, 'test.txt'))
            self.save_cache(path)

    def load_cache(self, path):
        for cache in ['dict.pt', 'train.pt', 'valid.pt', 'test.pt']:
            cache_path = os.path.join(path, cache)
            if not os.path.exists(cache_path):
                return False
        self.dictionary = torch.load(os.path.join(path, 'dict.pt'))
        self.train = torch.load(os.path.join(path, 'train.pt'))
        self.valid = torch.load(os.path.join(path, 'valid.pt'))
        self.test = torch.load(os.path.join(path, 'test.pt'))
        return True

    def save_cache(self, path):
        torch.save(self.dictionary, os.path.join(path, 'dict.pt'))
        torch.save(self.train, os.path.join(path, 'train.pt'))
        torch.save(self.valid, os.path.join(path, 'valid.pt'))
        torch.save(self.test, os.path.join(path, 'test.pt'))
    
    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

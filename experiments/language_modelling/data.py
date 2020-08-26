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
    def __init__(self, path, train_frac=1.0, voc_pad=0, full_test=False):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

        # pad dictionary with <unk0>, <unk1>, etc, up to pad size.
        for i in range(voc_pad):
            self.dictionary.add_word(f'<unk{i}>')

        # select only the specified fraction 'train_frac' of data.
        new_train_count = int(train_frac * self.train.size(0))
        new_valid_count = int(train_frac * self.valid.size(0))
        new_test_count = int(train_frac * self.test.size(0))
        self.train = self.train[:new_train_count]
        self.valid = self.valid[:new_valid_count]
        if not full_test: self.test = self.test[:new_test_count]

        print(f"Number of tokens in each set({train_frac*100}% of total training data):")
        print("\ttrain: {} \n\tvalid: {}\n\ttest: {}"
              .format(self.train.size(0), self.valid.size(0), self.test.size(0)))

    def tokenize(self, path, padding=0):
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

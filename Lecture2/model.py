# -*- coidng utf-8 -*-
import torch.nn as nn
from torch.autograd import Variable
from collections import Counter, defaultdict, deque


class WORD2VEC(nn.Module):
    def __init__(self, N, half_window_size, lr, mode='cbow'):
        """
        V: vocab_size
        N: hidden layer size(word vector size)
        window_size: how many words that you want to see near target word
        mode: cbow / skipgram
        """
        super(WORD2VEC, self).__init__()

        self.V = None
        self.N = N
        # vocab and data setting
        self.half_window_size = half_window_size
        self.vocab_count = Counter()
        self.vocab2idx = defaultdict()
        self.vocab2idx['NULL'] = 0
        self.lr = lr

    def build_network(self):
        # network setting
        self.i2h = nn.Embedding(self.V, self.N, padding_idx=0)  # Embedding
        self.h2o = nn.Linear(self.N, self.V)
        self.softmax = nn.Softmax(dim=1)

    def get_vocabulary(self, corpus_list):
        for sentence in corpus_list:
            self.vocab_count.update(sentence)
        for i, w in enumerate(self.vocab_count.keys()):
            self.vocab2idx[w] = i + 1
        self.idx2vocab = {i: w for w, i in self.vocab2idx.items()}

    def generate_batch(self, sentence):
        # sentence size와 window size 결정 조건 확인(추가할것)
        target_words = []
        batch_windows = []

        # add padding data
        batch_sentence = ['NULL'] * self.half_window_size + sentence + ['NULL'] * self.half_window_size
        for i, target_word in enumerate(sentence):
            target_words.append(target_word)
            center_idx = i + self.half_window_size
            window = deque(maxlen=self.half_window_size * 2)
            window.extendleft(reversed(batch_sentence[i:center_idx]))
            window.extend(batch_sentence[center_idx + 1:center_idx + 1 + self.half_window_size])
            batch_windows.append(window)

        return batch_windows, target_words

    def data_transfer(self, corpus_list):
        batch_X = []
        batch_y = []
        for sentence in corpus_list:
            batch_windows, target_words = self.generate_batch(sentence)
            for window in batch_windows:
                batch_X.append([self.vocab2idx[word] for word in window])
            for target in target_words:
                batch_y.append([self.vocab2idx[target]])
        return batch_X, batch_y

    def fit(self, corpus):
        corpus_list = [sentence.split() for sentence in corpus]
        self.get_vocabulary(corpus_list)
        self.V = len(self.vocab2idx)
        X, y = self.data_transfer(corpus_list)
        self.build_network()
        print('fit done!')
        return X, y

    def forward(self, X):
        embed = self.i2h(X)
        h = Variable(embed.data.mean(dim=0).unsqueeze(0))
        output = self.h2o(h)
        probs = self.softmax(output)
        return output, probs
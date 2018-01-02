# -*- coidng utf-8 -*-
import torch.nn as nn
from torch.autograd import Variable
from collections import Counter, defaultdict, deque
from nltk.tokenize import word_tokenize


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
        """batch_data = [windows(list), target(list)]"""
        batch_data = []
        for sentence in corpus_list:
            batch_windows, target_words = self.generate_batch(sentence)
            for window, target in zip(batch_windows, target_words):
                idxed_window = [self.vocab2idx[word] for word in window]
                idxed_target = [self.vocab2idx[target]]
                batch_data.append([idxed_window, idxed_target])
        return batch_data

    def tokenize_corpus(self, corpus):
        """문장에 부호를 제거하고 단어 단위로 tokenize 한다"""
        check = ['.', '!', ':', ',', '(', ')', '?', '@', '#', '[', ']', '-', '+', '=', '_']
        corpus_list = []
        for sentence in corpus:
            temp = word_tokenize(sentence)
            temp = [word.lower() for word in temp if word not in check]
            corpus_list.append(temp)
        return corpus_list
    
    def fit(self, corpus):
        """
        corpus를 학습시킬 데이터로 전환시켜준다. 모든 데이터는 단어의 vocab2idx를 근거해서 바뀐다.
        Vocab이 설정되면 네트워크도 같이 설정된다.
        batch_data = [window, target]
        """
        corpus_list = self.tokenize_corpus(corpus)
        self.get_vocabulary(corpus_list)
        self.V = len(self.vocab2idx)
        batch_data = self.data_transfer(corpus_list)
        self.build_network()
        print('fit done!')
        return batch_data

    def forward(self, X):
        embed = self.i2h(X)  # batch x V x N
        h = Variable(embed.data.mean(dim=1))  # batch x N
        output = self.h2o(h)  # batch x V
        probs = self.softmax(output)  # batch x V
        return output, probs
# -*- coidng utf-8 -*-
from model import WORD2VEC
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch
import matplotlib.pylab as plt
from operator import itemgetter
import pandas as pd
from scipy.spatial.distance import cosine


def create_sample_data():
    corpus = ['the king loves the queen',
              'the queen loves the king',
              'the dwarf hates the king',
              'the queen hates the dwarf',
              'the dwarf poisons the king',
              'the dwarf poisons the queen',]

    return corpus


def train(corpus, n_epoch, N, half_window_size, lr):
    word2vec = WORD2VEC(N=N, half_window_size=half_window_size, lr=lr)
    X, y = word2vec.fit(corpus)

    F = nn.CrossEntropyLoss()
    optimizer = optim.SGD(word2vec.parameters(), lr=word2vec.lr)

    loss_list = []
    for epoch in range(n_epoch):

        for batch_X, batch_y in zip(X, y):
            optimizer.zero_grad()
            batch_X = Variable(torch.LongTensor(batch_X))
            batch_y = Variable(torch.LongTensor(batch_y))

            output, probs = word2vec.forward(batch_X)

            loss = F(output, batch_y)

            loss.backward()
            optimizer.step()
        loss_list.append(loss.data[0])
        if epoch % 1000 == 0:
            print('#{}| loss:{}'.format(epoch, loss.data[0]))

    return word2vec, loss_list


def draw_loss_graph(loss_list, n_epoch):
    xx = np.linspace(0, n_epoch, num=n_epoch)
    plt.figure(figsize=(8, 8))
    plt.plot(xx, loss_list, label='loss')
    plt.legend(fontsize=20)
    plt.title('Loss', fontsize=30)
    plt.show()


def get_word_vector(word2vec, save_path=None):
    vector = word2vec.i2h.weight.data.numpy()

    idx_list = sorted([(w, i) for w, i in word2vec.vocab2idx.items()], key=itemgetter(1))
    idx_list = [t[0] for t in idx_list]
    df = pd.DataFrame(vector, index=idx_list).T

    if save_path:
        df.T.to_csv(save_path, sep='\t')

    return df, vector


def draw_vectors(df):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.scatter(df.values[0, 1:], df.values[1, 1:])
    ax.grid(False)
    ax.plot((df.values[0, 1:].max() + 0.2, df.values[0, 1:].min() - 0.2), (0, 0), 'k-', linewidth=0.5)
    ax.plot((0, 0), (df.values[1, 1:].max() + 0.2, df.values[1, 1:].min() - 0.2), 'k-', linewidth=0.5)
    plt.title('Word Vectors', fontsize=20)
    for i, txt in enumerate(df.columns[1:]):
        ax.annotate(txt, (df.values[0, 1:][i] - 0.02,
                          df.values[1, 1:][i] + 0.08))

    plt.show()


def cosine_similarity(word1, word2, df):
    v_1 = df[word1].values
    v_2 = df[word2].values
    return cosine(v_1, v_2)
from torch import nn
import os
import pickle as pkl
import torch
import numpy as np
from tqdm import tqdm
from .positional import PositionalEmbedding
import math


class TokenEmbedding(nn.Module):
    def __init__(self, args, vocab):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.args = args
        if self.args.pretrain:
            self.get_embedding()
            self.embedding = nn.Embedding.from_pretrained(self.embedding_matrix, padding_idx=self.vocab.pad_index,
                                                          freeze=False)
        else:
            # self.embedding_dim = self.args.hidden
            self.embedding = nn.Embedding(self.vocab_size, self.args.hidden, padding_idx=self.vocab.pad_index)
        # self.linear = nn.Linear(self.embedding_dim,
        #                         self.args.hidden) if self.embedding_dim != self.args.hidden else None

    def get_embedding(self):
        embedding_dir = './catch/{}'.format(self.args.dataset)
        if not os.path.exists(embedding_dir):
            os.makedirs(os.path.join(embedding_dir))
        embedding_path = './catch/{}/{}_embedding.pkl'.format(self.args.dataset, self.vocab.type)
        if os.path.exists(embedding_path):
            with open(embedding_path, 'rb') as f:
                embedding = pkl.load(f)
                print('Load Embedding from Catch')
                self.embedding_matrix = embedding.clone().detach()
                assert len(embedding) == self.vocab_size
                # self.embedding_dim = embedding.shape[-1]
        else:
            with open(self.args.embedding_file, 'r') as f:
                line = f.readline().strip().split()
                self.embedding_dim = len(line) - 1
            # self.embedding_matrix = torch.randn(self.vocab_size, self.embedding_dim)
            self.embedding_matrix = torch.randn(self.vocab_size, self.args.hidden)
            count = 0
            with open(self.args.embedding_file, 'r', encoding='utf-8') as f:
                print('Create {} Embedding from raw data'.format(self.vocab.type))
                lines = f.readlines()
                for line in tqdm(lines):
                    line = line.strip().split()
                    word = line[0]
                    idx = self.vocab.find(word)
                    if idx != self.vocab.unk_index:
                        vector = torch.tensor(np.append(np.array(line[1:]),
                                                        np.array([0] * (self.args.hidden - self.embedding_dim))).astype(
                            np.float))
                        self.embedding_matrix[idx] = vector
                        count += 1
                print('Pretrain Word = {} for {}'.format((count / self.vocab_size), self.vocab.type))
            with open(embedding_path, 'wb') as f:
                pkl.dump(self.embedding_matrix, f)
            print('Save {} Embedding into catch'.format(self.vocab.type))


class LeftEmbedding(TokenEmbedding):
    def __init__(self, args, vocab):
        super().__init__(args, vocab)
        self.p = PositionalEmbedding(args.hidden, args.max_code_length)

    def forward(self, content):
        '''

        :param content: bs,max_code_length
        :param content_mask: bs,max_code_length
        :return:bs,max_code_length,hidden
        '''
        c_1 = self.embedding(content)
        # if self.linear:
        #     c_1 = self.linear(c_1)
        # bs,max_code_length,hidden

        # c_2 = c_1.masked_fill(content_mask.unsqueeze(-1) == 0, 0.0)
        # c_3 = torch.sum(c_2, dim=-2)  # bs,max_code_length,hidden
        return c_1 + self.p(c_1)


class RightEmbedding(TokenEmbedding):
    def __init__(self, args, vocab):
        super().__init__(args, vocab)
        self.out = nn.Linear(self.args.hidden, self.vocab_size)
        self.p = PositionalEmbedding(args.hidden, args.max_target_len)

    def forward(self, f_source):
        '''

        :param f_source: bs,max_target_len
        :return:bs,max_target_len,hidden
        '''
        c_1 = self.embedding(f_source)
        # if self.linear:
        #     c_1 = self.linear(c_1)
        # bs,max_target_len,hidden
        return c_1 + self.p(c_1)

    def prob(self, data):
        return self.out(data)

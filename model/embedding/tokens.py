from torch import nn
import os
import pickle as pkl
import torch
import numpy as np
from tqdm import tqdm


class TokenEmbedding(nn.Module):
    def __init__(self, args, vocab):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.args = args
        if self.args.pretrain:
            self.get_embedding()
            self.embedding = nn.Embedding.from_pretrained(self.embedding_matrix, padding_idx=self.vocab.pad_index)
        else:
            self.embedding_dim = self.args.hidden
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.vocab.pad_index)
        self.linear = nn.Linear(self.embedding_dim,
                                self.args.hidden) if self.embedding_dim != self.args.hidden else None

    def get_embedding(self):
        embedding_path = './catch/{}_embedding.pkl'.format(self.vocab.type)
        if os.path.exists(embedding_path):
            with open(embedding_path, 'rb') as f:
                embedding = pkl.load(f)
                print('Load Embedding from Catch')
                self.embedding_matrix = torch.tensor(embedding)
                assert len(embedding) == self.vocab_size
                self.embedding_dim = embedding.shape[-1]
        else:
            with open(self.args.embedding_file, 'r') as f:
                line = f.readline().strip()
                self.embedding_dim = len(line) - 1
            self.embedding_matrix = torch.randn(self.vocab_size, self.embedding_dim)
            count = 0
            with open(self.args.embedding_file, 'r') as f:
                print('Create {} Embedding from raw data'.format(self.vocab.type))
                lines = f.readlines()
                for line in tqdm(lines):
                    line = line.strip()
                    word = line[0]
                    vector = torch.tensor(np.array(line[1:]).astype(np.float))
                    idx = self.vocab.find(word)
                    if idx != self.vocab.unk_index:
                        self.embedding_matrix[idx] = vector
                        count += 1
                print('Pretrain Word = {} for {}'.format((count / self.vocab_size), self.vocab.type))
            with open(embedding_path, 'wb') as f:
                pkl.dump(self.embedding_matrix, f)
            print('Save {} Embedding into catch'.format(self.vocab.type))


class LeftEmbedding(TokenEmbedding):
    def __init__(self, args, vocab):
        super().__init__(args, vocab)
        assert self.vocab.type == 'source'

    def forward(self, content, content_mask):
        '''

        :param content: bs,max_code_length,sub_token_length
        :param content_mask: bs,max_code_length,sub_token_length
        :return:bs,max_code_length,hidden
        '''
        c_1 = self.embedding(content)
        if self.linear:
            c_1 = self.linear(c_1)
        # bs,max_code_length,sub_token_length,hidden

        c_2 = c_1.masked_fill(content_mask.unsqueeze(-1) == 0, 0.0)
        c_3 = torch.sum(c_2, dim=-2)  # bs,max_code_length,hidden
        return c_3


class RightEmbedding(TokenEmbedding):
    def __init__(self, args, vocab):
        super().__init__(args, vocab)
        assert self.vocab.type == 'target'
        self.out = nn.Linear(self.args.hidden, self.vocab_size)

    def forward(self, f_source):
        '''

        :param f_source: bs,max_target_len
        :return:bs,max_target_len,hidden
        '''
        c_1 = self.embedding(f_source)
        if self.linear:
            c_1 = self.linear(c_1)
        # bs,max_target_len,hidden
        return c_1

    def prob(self, data):
        return self.out(data)

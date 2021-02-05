import torch.nn as nn
import torch.nn.functional as F
import torch

import math
from .single import RelationAwareAttention


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = RelationAwareAttention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, relation=None, mask=None):
        '''

        :param query: bs, max_code_length, hidden
        :param key: bs, max_code_length, hidden
        :param value: bs, max_code_length, hidden
        :param relation: bs, max_code_length,max_code_length, hidden
        :param mask:bs, 1,max_code_length,max_code_length
        :return:
        '''
        batch_size, max_code_length = query.size(0), query.size(1)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        if relation is not None:
            relation_k, relation_v = [l(x).view(batch_size, max_code_length, max_code_length, self.h,
                                                self.d_k).transpose(2, 3).transpose(1, 2) for l, x in
                                      zip([self.linear_layers[1], self.linear_layers[2]], (relation, relation))]
        else:
            relation_k, relation_v = None, None

        x, attn = self.attention(query, key, value, relation_k, relation_v, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

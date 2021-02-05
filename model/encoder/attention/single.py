import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class RelationAwareAttention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, relation_k=None, relation_v=None, mask=None, dropout=None):
        '''

        :param query: bs, head,max_code_length, hidden//head
        :param key:
        :param value:
        :param relation_k: bs, head,max_code_length,max_code_length, hidden//head
        :param relation_v: bs, head,max_code_length,max_code_length, hidden//head
        :param mask:bs, 1,max_code_length,max_code_length
        :param dropout:
        :return:
        #  a*b  (a*b, a*a*b)
        #  b * (a*b a*b)
        #  a*b  * (1*a*b   a*a*b)
        #  'ij,kj->ik'
        #  'ij,ikj->ik'
        #  然后加起来等于ik

        #  'ik,kj->ij'
        #  'ik,ikj->ij'
        # 然后加起来 就等于value
        # 所以关键问题是定出ikj这个relation
        '''
        # TODO
        score = torch.einsum('bhik,bhjk->bhij', query, key)

        if relation_k is not None:
            score_r = torch.einsum('bhik,bhijk->bhij', query, relation_k)
            score += score_r

        scores = score / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        attn_sum = torch.einsum('bhij,bhjk->bhik', p_attn, value)

        if relation_v is not None:
            r_attn_sum = torch.einsum('bhij,bhijk->bhik', p_attn, relation_v)
            attn_sum += r_attn_sum

        return attn_sum, p_attn

import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class RelationAwareAttention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, relation=None, path_map=None, mask=None, dropout=None):
        '''

        :param path_map: bs,max_code_length,max_code_length
        :param query: bs, head,max_code_length, hidden//head
        :param key:
        :param value:
        :param relation: bs,max_path_num+1,hidden//head
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
        bs, h, max_code_length, dim = query.shape

        if relation is not None:
            bs, max_path_num, dim = relation.shape
            # relation_k: bs,h,(max_path_num+1),dim
            score_r = torch.matmul(query, relation.unsqueeze(1).transpose(-1, -2)).gather(-1, path_map.unsqueeze(1))
            score += score_r  # TODO 这个地方可以考虑加上一个dropout
        # TODO xl式的相对位置编码
        scores = score / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        attn_sum = torch.einsum('bhij,bhjk->bhik', p_attn, value)

        if relation is not None:
            bs, max_path_num, dim = relation.shape
            r_attn_sum = torch.zeros(bs, h, max_code_length, max_path_num).to(relation.device). \
                scatter_add_(-1, path_map.unsqueeze(1).expand(-1, h, -1, -1), p_attn).matmul(relation.unsqueeze(1))
            attn_sum += r_attn_sum

        return attn_sum, p_attn

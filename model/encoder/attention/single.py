import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class RelationAwareAttention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, relation_k=None, relation_v=None, path_map=None, mask=None, dropout=None):
        '''

        :param path_map: bs,max_code_length,max_code_length
        :param query: bs, head,max_code_length, hidden//head
        :param key:
        :param value:
        :param relation_k: bs,h,max_path_num,dim
        :param relation_v: bs,h,max_path_num,dim
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
        bs, max_code_length = query.shape[0], query.shape[2]

        if relation_k is not None:
            bs, h, max_path_num, dim = relation_k.shape

            # bs,h,max_path_num,dim
            # bs,h,max_code_length,dim
            # bs,max_code_length,max_code_length

            # 1)bs,h,max_path_num,dim and bs,h,max_code_length,max_code_length
            # -> bs,h,max_code_length,max_code_length,dim

            relation_k = torch.cat((relation_k, torch.zeros(1, 1, 1, 1).expand(bs, h, -1, dim).to(relation_k.device)),
                                   dim=2)
            # relation_k: bs,h,(max_path_num+1),dim

            path_map_ = path_map.masked_fill(path_map == -1, max_path_num).unsqueeze(1).expand(-1, h, -1, -1)
            # path_map: bs,h,max_code_length,max_code_length

            output = torch.gather(relation_k.unsqueeze(2).expand(-1, -1, max_code_length, -1, -1), 3,
                                  path_map_.unsqueeze(-1).expand(-1, -1, -1, -1, dim))
            # relation_k: bs,h,max_code_length,(max_path_num+1),dim
            # path_map: bs,h,max_code_length,max_code_length,dim
            # output:bs,h,max_code_length,max_code_length,dim

            score_r = torch.einsum('bhik,bhijk->bhij', query, output)
            score += score_r

        scores = score / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        attn_sum = torch.einsum('bhij,bhjk->bhik', p_attn, value)

        if relation_v is not None:
            bs, h, max_path_num, dim = relation_v.shape

            # bs,h,max_path_num,dim
            # bs,h,max_code_length,dim
            # bs,max_code_length,max_code_length

            # 1)bs,h,max_path_num,dim and bs,h,max_code_length,max_code_length
            # -> bs,h,max_code_length,max_code_length,dim

            relation_v = torch.cat((relation_v, torch.zeros(1, 1, 1, 1).expand(bs, h, -1, dim).to(relation_v.device)),
                                   dim=2)
            # relation_k: bs,h,(max_path_num+1),dim

            path_map_ = path_map.masked_fill(path_map == -1, max_path_num).unsqueeze(1).expand(-1, h, -1, -1)
            # path_map: bs,h,max_code_length,max_code_length

            output = torch.gather(relation_v.unsqueeze(2).expand(-1, -1, max_code_length, -1, -1), 3,
                                  path_map_.unsqueeze(-1).expand(-1, -1, -1, -1, dim))
            # relation_k: bs,h,max_code_length,(max_path_num+1),dim
            # path_map: bs,h,max_code_length,max_code_length,dim
            # output:bs,h,max_code_length,max_code_length,dim

            r_attn_sum = torch.einsum('bhij,bhijk->bhik', p_attn, output)
            attn_sum += r_attn_sum

        return attn_sum, p_attn

from torch import nn
import os
import torch


class PathEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = nn.Embedding(self.args.path_embedding_num + 1, self.args.path_embedding_size,
                                      padding_idx=self.args.path_embedding_num)
        self.rnn = nn.GRU(self.args.path_embedding_size, self.args.hidden, batch_first=True)

    def forward(self, paths, paths_mask, path_map):
        '''

        :param paths: bs,2*max_path_num,max_path_length
        :param paths_mask: bs,2*max_path_num
        :param path_map: bs,max_code_length,max_code_length
        :return:bs,2*max_path_num,hidden
        '''
        p_1 = self.embedding(paths)
        # bs,2*max_path_num,max_path_length,dim

        p_2 = p_1.view(-1, p_1.shape[-2], p_1.shape[-1])
        # bs*2*max_path_num,max_path_length,dim

        m_1 = paths_mask.view(-1)
        # bs*2*max_path_num

        m_2, m_2_idx = torch.sort(m_1, descending=True)
        _, re_m_2_idx = torch.sort(m_2_idx)
        p_3 = p_2.index_select(0, m_2_idx)
        p_4 = torch.nn.utils.rnn.pack_padded_sequence(p_3, m_2.cpu(), batch_first=True)
        _, p_5 = self.rnn(p_4)
        p_6 = p_5.squeeze(0)  # bs*2*max_path_num,max_path_length,hidden
        p_7 = p_6.index_select(0, re_m_2_idx)  # 这个地方不对劲
        # bs*2*max_path_num,hidden

        temp_p_7 = torch.cat((p_7, torch.zeros(1, p_7.shape[-1]).to(p_7.device)), dim=0)
        # bs*2*max_path_num+1,hidden
        # cat a zero tensor to index

        p_m_1 = path_map.view(-1)
        # bs*max_code_length*max_code_length

        p_m_2 = p_m_1.masked_fill(p_m_1 == -1, p_7.shape[0])
        # change -1 to max num, get zero tensor

        p_8 = torch.index_select(temp_p_7, 0, p_m_2)
        # bs*max_code_length*max_code_length,hidden

        p_9 = p_8.view(path_map.shape[0], path_map.shape[1], path_map.shape[2], -1)
        # bs,max_code_length,max_code_length,hidden

        return p_9
